import math
import shutil
import random
import argparse
import numpy as np
import cv2
import imageio.v2 as imageio
import os, subprocess
from _gradient import get_gradient

import time

def draw_order(h, w, scale):
    order = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-scale // 2, scale // 2) + i
            x = random.randint(-scale // 2, scale // 2) + j
            order.append((y % h, x % w))
    return order

def main(args):

    brush_width = int(args.brush_width)
    quant = float(args.palette)    # Color come from palette(limited colours)

    img_path = args.path.rsplit(".", -1)
    img_name = ".".join(img_path[:len(img_path) - 1])
    img_extension = img_path[len(img_path) - 1]

    if img_extension == "gif":
        temp_input_name = "temp.gif"
    else:
        temp_input_name = "temp.jpg"
    temp_input_path = os.path.join(os.getcwd(), temp_input_name)
    shutil.copy(args.path, temp_input_path)

    if img_extension == 'gif': # convert gif into jpgs
        shutil.rmtree(img_name, ignore_errors = True) # in case this has been done with a previous gif before
        os.makedirs(img_name)
        subprocess.call(('convert -verbose -coalesce %s %s/no.jpg' % (args.path, img_name)).split(), stdout = subprocess.DEVNULL)
        all_img_path = [(img_name+'/no-'+str(i)+'.jpg') for i in range(len(os.listdir(img_name)))]
    else:
        all_img_path = [args.path]

    # Process the image / gif
    result = []
    print("Frames of the gif exported to:", all_img_path)
    for path in all_img_path:
        print("\nCurrently running:" + path)
        img = cv2.imread(path)
        
        # Get the gradient of image (Using sobel, scharr, prewitt, or roberts)
        print("Gradient: "); s = time.time()
        r = 2 * int(img.shape[0] / 50) + 1
        Gx, Gy = get_gradient(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (r, r), args.gradient)
        Gh = np.sqrt(np.sqrt(np.square(Gx) + np.square(Gy)))    # Length of the ellipse
        Ga = (np.arctan2(Gy, Gx) / np.pi) * 180 + 90            # Angle of the ellipse
        print("%.4f secs." % float(time.time() - s))

        print("Drawing"); s = time.time()
        canvas = cv2.medianBlur(img, 11)    # Make the image artistic
        order = draw_order(img.shape[0], img.shape[1], scale=brush_width*2)

        # Draw the ellipse
        colors = np.array(img, dtype=float)
        for i, (y, x) in enumerate(order):
            length = int(round(brush_width + brush_width * Gh[y, x]))
            # Select color
            if quant != 0: color = np.array([round(colors[y,x][0]/quant)*quant+random.randint(-5,5), 
                round(colors[y,x][1]/quant)*quant+random.randint(-5,5), round(colors[y,x][2]/quant)*quant+random.randint(-5,5)], dtype=float)
            else: color = colors[y,x]

            cv2.ellipse(canvas, (x, y), (length, brush_width), Ga[y, x], 0, 360, color, -1, cv2.LINE_AA)

        result.append(canvas)
        print("%.4f secs." % float(time.time() - s))

    # Output the result
    print("\nOutput the result")
    output_path = temp_input_path

    if img_extension == 'gif':
        c, images = 0, []
        for canva in result:
            cv2.imwrite(img_name+'/r-'+str(c)+'.jpg', canva); c += 1
        for i in range(c-1):
            images.append(imageio.imread(img_name+'/r-'+str(i)+'.jpg'))

        from PIL import Image
        imageio.mimsave(output_path, images, duration=float(Image.open(args.path).info['duration']) / 1000, loop = 0)
        shutil.rmtree(img_name, ignore_errors=True)
    else:
        cv2.imwrite(output_path, result[0])

    shutil.copy(temp_input_path, args.output_path)
    os.remove(temp_input_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--brush_width', default=5, help="Scale of the brush strokes")
    parser.add_argument('--path', required=True, type=str, help="Target image path, gif of still image")
    parser.add_argument('--palette', default=0, help="Palette colours. 0 = Actual color")
    parser.add_argument('--gradient', default='sobel', help="Edge detection type. (sobel, scharr, prewitt, roberts)")
    parser.add_argument('--output_path', required=True, help="Output path. File extension added automatically.")
    args = parser.parse_args()
    main(args)
