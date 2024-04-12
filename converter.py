import os

import shlex

from PIL import Image, ImageOps
from dithering import ordered_dither
import numpy as np
import cv2

import math

from glitch_this import ImageGlitcher
glitcher = ImageGlitcher()

# input_path = "/home/account/Documents/writ/if/shufflecomp/crumbling_castle/crumbling_castle_source/export/assets/image_archive/conversion"
current_path = os.path.abspath(os.getcwd())
input_path = current_path

def rename(i, output_num, input_num):
    if str(input_num) != "-1" and not os.path.basename(i).startswith(str(input_num)):
        return

    inp = Image.open(i)
    inp.save(filename(i, output_num))

def filename(i, num):
    i = os.path.basename(i)
    return str(num) + i[1:-4] + ".png"

def posterize(i, output_num, input_num):
    if not os.path.basename(i).startswith(str(input_num)):
        return

    inp = Image.open(i)
    inp = ImageOps.posterize(inp, 3)
    inp.save(filename(i, output_num))

def dither(i, output_num, input_num, keep_transparency = True):
    if not os.path.basename(i).startswith(str(input_num)):
        return

    inp = cv2.imread(i)
    inp = ordered_dither(inp, "Bayer2x2")
    output_name = filename(i, output_num)
    cv2.imwrite(output_name, inp)

    if keep_transparency:
        base_img = Image.open(i).convert("RGBA")
        overlay_img = Image.open(output_name).convert("RGBA")
        alpha = base_img.split()[-1]
        mask_image = Image.new("RGBA", base_img.size, (0, 0, 0, 255))
        mask_image.paste(alpha, mask=alpha)
        mask_image = mask_image.convert('L')

        overlay_img.putalpha(mask_image)
        overlay_img.save(output_name)
        return
        real_output = Image.composite(base_img, overlay_img, mask_image)
        real_output.save(output_name)

def resize(i, output_num, input_num, min_height = 804, max_height = 850):
    if not os.path.basename(i).startswith(str(input_num)):
        return

    inp = Image.open(i)
    if (inp.height < min_height):
        factor = min_height/inp.height
        inp = inp.resize((math.ceil(inp.width * factor), math.ceil(inp.height * factor)), resample=Image.Resampling.BILINEAR)

    if (inp.height > max_height):
        factor = max_height/inp.height
        inp = inp.resize((math.ceil(inp.width * factor), math.ceil(inp.height * factor)), resample=Image.Resampling.NEAREST)

    inp.save(filename(i, output_num))

# pixelate
def pypxl(i, output_num, input_num):
    if not os.path.basename(i).startswith(str(input_num)):
        return

    input_path2 = os.path.join(current_path, "pypxl")
    os.chdir(input_path2)

    command = "python3 pypxl_image.py -s 512 512 " + os.path.join(input_path, i) + " " + os.path.join(input_path, filename(i, output_num))
    os.system(command)

    os.chdir(current_path)

# easier to use a screenshot with https://coolors.co/gradient-maker
def gen_colormaps():
    colormap_name = "colormap.png"

    colors = ["rgb(0, 50, 100)", "rgb(220, 110, 110)", "rgb(255, 255, 0)", "rgb(255, 255, 255)"]
    sizes = [160, 60, 35]
    command = ["convert "]

    for i in range(len(sizes)):
        command.append("-size " + str(sizes[i]) + "x1 gradient:'" + colors[i] + "'-'" + colors[i+1] + "' ")

    command.append("+append colormap.png")
    command = "".join(command)
    os.system(command)

    return colormap_name

def apply_colormap(i, output_num, input_num, colormap_name, gradient_alpha):
    if not os.path.basename(i).startswith(str(input_num)):
        return

    command = "convert " + i + \
        " -channel RGB -separate -evaluate-sequence mean " +  \
        colormap_name + \
        " -clut " +  \
        "temp.png"
    os.system(command)

    base_img = Image.open(i)
    overlay_img = Image.open("temp.png").convert("RGBA")
    overlay_img.putalpha(gradient_alpha)
    base_img.paste(overlay_img, mask=overlay_img)
    base_img.save(filename(i, output_num))

def glitch(i, output_num, input_num):
    if not os.path.basename(i).startswith(str(input_num)):
        return

    # glitch_image(self, src_img, glitch_amount, glitch_change=0.0, cycle=False, color_offset=False, scan_lines=False, gif=False, frames=23, step=1)
    glitch_img = glitcher.glitch_image(i, 0.7, color_offset = True, gif=True)
    # glitch_img = glitcher.glitch_image(i, 0.5, scan_lines = True, gif=True)

    DURATION = 200      # Set this to however many centiseconds each frame should be visible for
    LOOP = 0            # Set this to how many times the gif should loop
                        # LOOP = 0 means infinite loop
    glitch_img[0].save(filename(i, output_num) + ".gif",
                   format='GIF',
                   append_images=glitch_img[1:],
                   save_all=True,
                   duration=DURATION,
                   loop=LOOP)

if __name__ == "__main__":
    # name = gen_colormaps()
    for i in [x for x in sorted(os.listdir(input_path)) if x.endswith(".png")]:
        if "tunnel" in i:
            print(i)
            i = os.path.join(input_path, i)
            # rename(i, 1, 0)
            pypxl(i, 1, 0)
            resize(i, 2, 1, max_height = 350)
            # dither(i, 2, 1)
            apply_colormap(i, 3, 2, "colormap_glass.png", 190)
            resize(i, 4, 3, min_height = 850)
            glitch(i, 5, 4)
            pass

    # i = "1dorm.png"
    # pypxl(i, 1, 0)
    # glitch(i, 1, 1)
