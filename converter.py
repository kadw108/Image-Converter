import os
import subprocess
import shlex

from PIL import Image, ImageOps, ImageEnhance
from dithering import ordered_dither
import numpy as np
import cv2
import math

from glitch_this import ImageGlitcher
glitcher = ImageGlitcher()

import re
import shutil
from natsort import natsorted

import pathlib

current_path = os.path.abspath(os.getcwd())
input_path = current_path

""" --- IMAGE OR GIF ALTERING FUNCTIONS --- """
""" 1. FUNCTIONS THAT ALTER STATIC IMAGES """

def posterize(filename, output_name, keep_transparency = True, bits = 2):
    """
    Posterize input png.

    bits = The number of bits to keep for each channel (1-8) (less = more posterized)
    """

    original_img = Image.open(filename).convert("RGBA")

    inp = Image.open(filename).convert("RGB")
    inp = ImageOps.posterize(inp, bits)
    inp.save(output_name)

    if keep_transparency:
        overlay_img = Image.open(output_name).convert("RGBA")
        alpha = original_img.split()[-1]
        mask_image = Image.new("RGBA", original_img.size, (0, 0, 0, 255))
        mask_image.paste(alpha, mask=alpha)
        mask_image = mask_image.convert('L')

        overlay_img.putalpha(mask_image)
        overlay_img.save(output_name)

def dither(filename, output_name, keep_transparency = True, strength = 255):
    """
    Dither input png.

    keep_transparency -- whether to preserve alpha in input png
    strength -- strength of dither, i.e. how opaque the dithered version of the image will be over the original image
    """

    original_img = Image.open(filename).convert("RGBA")

    inp = cv2.imread(filename)
    inp = ordered_dither(inp, "Bayer2x2")
    cv2.imwrite(output_name, inp)
  
    if strength != 255:
        overlay_img = Image.open(output_name).convert("RGBA")

        base_img = original_img.copy().convert("RGB")
        overlay_img.putalpha(strength)
        base_img.paste(overlay_img, mask=overlay_img)
        base_img.save(output_name)

    if keep_transparency:
        overlay_img = Image.open(output_name).convert("RGBA")
        alpha = original_img.split()[-1]
        mask_image = Image.new("RGBA", original_img.size, (0, 0, 0, 255))
        mask_image.paste(alpha, mask=alpha)
        mask_image = mask_image.convert('L')

        overlay_img.putalpha(mask_image)
        overlay_img.save(output_name)

def resize(filename, output_name, min_height = 804, max_height = 850):
    """
    Resize input png.
    """
    inp = Image.open(filename)
    if (inp.height < min_height):
        factor = min_height/inp.height
        inp = inp.resize((math.ceil(inp.width * factor), math.ceil(inp.height * factor)), resample=Image.Resampling.BILINEAR)

    if (inp.height > max_height):
        factor = max_height/inp.height
        inp = inp.resize((math.ceil(inp.width * factor), math.ceil(inp.height * factor)), resample=Image.Resampling.NEAREST)

    inp.save(output_name)

def crop(filename, output_name, max_width = 1920, max_height = 1080):
    """
    Crops input png if it has width/height greater
    than max_width/max_height.

    Crops sides of image (anchor in center).
    """
    inp = Image.open(filename)

    width, height = inp.size
    left = 0
    top = 0
    right = width
    bottom = height

    if (width > max_width):
        difference = width - max_width
        left_diff = math.ceil(difference/2)
        right_diff = math.floor(difference/2)
        left = left_diff
        right = width - right_diff

    if (height > max_height):
        difference = height - max_height
        top_diff = math.ceil(difference/2)
        bottom_diff = math.floor(difference/2)
        top = top_diff
        bottom = height - bottom_diff

    inp = inp.crop((left, top, right, bottom))
    inp.save(output_name)

# pixelate
def pypxl(filename, output_name):
    """
    Apply pixelate effect to input png, using pypxl library. Takes a bit longer than most of the functions here.
    """
    input_path2 = os.path.join(current_path, "pypxl")
    os.chdir(input_path2)

    command = "python3 pypxl_image.py -s 512 512 " + os.path.join(input_path, filename) + " " + os.path.join(input_path, output_name)
    os.system(command)

    os.chdir(current_path)

def apply_colormap(filename, output_name, colormap_name, gradient_alpha = 255, keep_transparency = True):
    """
    from https://stackoverflow.com/a/71584672

    Applies gradient map/colormap to input png.

    colormap_name -- path to colormap, 1d horizontal pixel strip with lighter colors to the right
    gradient_alpha -- strength of colormap; number from [0, 255] with higher = stronger colormap
    keep_transparency -- whether to preserve alpha in input png
    """

    original_img = Image.open(filename).convert("RGBA")

    im = Image.open(filename).convert('RGB')
    na = np.array(im)
    grey = np.mean(na, axis=2).astype(np.uint8)

    # Load colourmap
    cmap = Image.open(colormap_name).convert('RGB')
    # need to resize to 256 x 1 pixels
    # the below line did not exist in the original stackoverflow answer so I spent an hour trying to figure out why the colormaps weren't working properly
    cmap = cmap.resize((256, 1))

    # Make output image, same height and width as grey image, but 3-channel RGB
    result = np.zeros((*grey.shape, 3), dtype=np.uint8)

    np.take(cmap.getdata(), grey, axis=0, out=result)

    # overlay_img has map applied in full
    overlay_img = Image.fromarray(result).convert("RGB")
    overlay_img.save(output_name)
    base_img = Image.open(filename).convert("RGB")
    overlay_img.putalpha(gradient_alpha)
    base_img.paste(overlay_img, mask=overlay_img)
    base_img.save(output_name)

    if keep_transparency:
        overlay_img = Image.open(output_name).convert("RGBA")
        alpha = original_img.split()[-1]
        mask_image = Image.new("RGBA", original_img.size, (0, 0, 0, 255))
        mask_image.paste(alpha, mask=alpha)
        mask_image = mask_image.convert('L')

        overlay_img.putalpha(mask_image)
        overlay_img.save(output_name)

def pillow_adjust_image(filename, output_name, new_saturation = 1.0, new_contrast = 1.0, new_brightness = 1.0, new_sharpness = 1.0):
    """
    https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html

    New saturation/contrast/brightness/sharpness starts at 0, 1.0 is normal, 2.0 is double, etc
    """
    inp = Image.open(filename).convert("RGBA")
    inp = ImageEnhance.Color(inp).enhance(new_saturation)
    inp = ImageEnhance.Contrast(inp).enhance(new_contrast)
    inp = ImageEnhance.Brightness(inp).enhance(new_brightness)
    inp = ImageEnhance.Sharpness(inp).enhance(new_sharpness)
    inp.save(output_name)

""" 2. FUNCTIONS THAT TURN VIDEOS INTO IMAGE SEQUENCES """

def video_to_image_sequence(filename, output_name, fps = 10):
    """ 
    Turns input video into output png sequence.
    Outputs are named output_name_1.png, output_name_2.png, etc.

    fps = frames of the video to turn into pngs each second.
    """

    if not filename.endswith((".qt", ".mp4", ".mov")):
        return

    command = "ffmpeg -i " + filename + " -vf \"fps=" + str(fps) + "\" " + output_name + "_%d.png"
    subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell = True)

""" 3. FUNCTIONS THAT TURN FILES INTO GIFS """

def convert_gif(filename, output_name, number_frames = 8, framerate = 8): 
    """ 
    Turns input png into output gif.
    The gif will appear identical to the input png (it will not be animated).
    
    Also works on png sequences. If the input png ends with a number, e.g. string1.png, then any png starting with string will be incorporated into the resulting gif. To prevent creating the same gif twice, the gif will only be created if the input png sequence has a creation date newer than the resulting gif (if the resulting gif already exists).

    Useful for the painting function, which only converts pngs to pngs and gifs to gifs.

    number_frames = number of frames in resulting gif. Ignored with png sequence input.
    framerate = framerate of resulting gif.
    """

    if not filename.endswith((".png", ".jpg")):
        return

    # from https://stackoverflow.com/a/14471236
    filename_no_extension = pathlib.Path(filename).stem
    m = re.search(r'\d+$', filename_no_extension)
    # if the filename ends in digits, m will be a Match object; otherwise it will be None
    
    # if input is png sequence
    if m is not None:
        common_prefix = filename_no_extension[:-len(m.group())]
        frame_min = float("inf")
        frame_max = float("-inf")

        real_output_name = re.sub(r'\d+$', '', pathlib.Path(output_name).stem) + ".gif"
        output_gif_exists = os.path.exists(real_output_name)
        if output_gif_exists:
            output_gif_creation = os.path.getmtime(real_output_name)
            latest_sequence_creation = 0

        # get frame_min and frame_max
        for path in sorted(os.listdir(current_path)):
            if path.startswith(common_prefix):
                
                m = re.search(r'\d+$', pathlib.Path(path).stem)
                frame_num = int(m.group())
                
                frame_min = min(frame_min, frame_num)
                frame_max = max(frame_max, frame_num)
               
                if output_gif_exists and os.path.getmtime(path) > latest_sequence_creation:
                    latest_sequence_creation = os.path.getmtime(path)
    
        if not output_gif_exists or latest_sequence_creation > output_gif_creation:
            command = 'convert $(printf -- "-dispose Background "; for ((a=' + str(frame_min) + '; a<' + str(frame_max) + '; a++)); do printf -- "-delay ' + str(framerate) + ' ' + common_prefix + '%d.png -dispose Background " $a; done;) ' + real_output_name
           
            # From https://stackoverflow.com/a/29269316 
            # Normal subprocess.call or subprocess.run doesn't work here, possibly because of the nested commands
            process = subprocess.Popen('/bin/bash', stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            out, err = process.communicate(command) 
       
    # if input is single png file 
    else:
        shutil.rmtree("temp", ignore_errors = True) 
        os.makedirs("temp")
        for i in range(number_frames):
            shutil.copyfile(filename, os.path.join("temp", str(i) + ".png"))

        command = "ffmpeg -y -framerate " + str(framerate) + " -i temp/%d.png " + output_name + ".gif"
        subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell = True)

def glitch(filename, output_name, glitch_amount = 0.7):
    """
    Turns input png into glitchy gif.
    """
    # glitch_image(self, src_img, glitch_amount, glitch_change=0.0, cycle=False, color_offset=False, scan_lines=False, gif=False, frames=23, step=1)
    glitch_img = glitcher.glitch_image(filename, glitch_amount, color_offset = True, gif=True)
    # glitch_img = glitcher.glitch_image(i, 0.5, scan_lines = True, gif=True)

    DURATION = 200      # Set this to however many centiseconds each frame should be visible for
    LOOP = 0            # Set this to how many times the gif should loop
                        # LOOP = 0 means infinite loop
    glitch_img[0].save(output_name + ".gif",
                       format='GIF',
                       append_images=glitch_img[1:],
                       save_all=True,
                       duration=DURATION,
                       loop=LOOP)

""" 4. FUNCTIONS THAT EDIT GIFS """

def painting(filename, output_name, brush_width = 1, gradient = "scharr"):
    """
    From https://github.com/cyshih73/Faster-OilPainting with slight modifications

    Turns input gif into painting-like gif.
    (Or input png into painting-like png, maybe.)

    main.py [-h] [--brush_width BRUSH_WIDTH] --path PATH
        [--palette PALETTE] [--gradient GRADIENT]
    optional arguments:
      -h, --help            show this help message and exit
      --brush_width BRUSH_WIDTH
                            Scale of the brush strokes
      --path PATH           Target image path, gif or still image
      --output_path PATH    Output path (added by KADW)
      --palette PALETTE     Palette colours. 0 = Actual color
      --gradient GRADIENT   Edge detection type. (sobel, scharr, prewitt, roberts)

    NOT COMPATIBLE WITH IMAGE TRANSPARENCY.

    """

    input_path2 = os.path.join(current_path, "Faster-OilPainting")
    os.chdir(input_path2)

    command = "python3 main.py --brush_width " + str(brush_width) + " --path " + os.path.join(input_path, filename) + " --output_path " + os.path.join(input_path, output_name) + " --palette 0 --gradient " + gradient
    subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell = True)
    os.chdir(current_path)

def gif_downsize(filename, output_name, percent):
    """
    Reduces actual size of input gif, which also reduces file size.
    Requires ImageMagick to work on command line as convert.

    percent -- [0-100], percentage of original size to keep.
    """

    command = "convert -resize " + str(percent) + "% " + filename + " " + output_name
    os.system(command)

def gif_optimize(filename, output_name, lossy = 30, color_num = None):
    """
    Optimize input gif to reduce file size.
    Requires gifsicle to work on command line.

    color_num -- colorspace/number of colors in resulting gif, suggested values from 16 to 256. Optional. Number of colors will not be reduced if not included.
    lossy -- level of compression, suggested values from 30 to 200, with higher = more compression.
    """

    color_num_arg = ""
    if isinstance(color_num, int):
        color_num_arg = "--colors " + str(color_num)

    command = "gifsicle -O3 " + color_num_arg + " --lossy=" + str(lossy) + " -o " + output_name + " " + filename
    os.system(command)

def gif_change_speed(filename, output_name, fps = 12):
    """
    Change speed of input gif to fps.
    Requires ImageMagick to work on command line as convert.
    """
    argument = str(int(100/fps)) # Default fps of 12 makes this 8

    command = "convert -delay " + argument + " " + os.path.join(input_path, filename) + " " + os.path.join(input_path, output_name)
    os.system(command)

def gif_reduce_frames(filename, output_name, skip_frames=2):
    """
    Removes all frames except every [skip_frames] frame.
    Requires gifsicle to work on command line.

    skip_frames -- skip_frames=2 keeps frames 0, 2, 4... skip_frames=4 keeps frames 0, 4, 8... and so on.
    """

    im = Image.open(filename)
    num_frames = str(im.n_frames - 1) # frames are 0-indexed in gifsicle

    command = "gifsicle -U " + filename + " `seq -f \"#%g\" 0 " + str(skip_frames) + " " + num_frames + "` -o " + output_name
    os.system(command)

""" --- UTILITY FUNCTIONS --- """

def copy(filename, output_name):
    """
    Opens filename as image and saves it under output_name.
    Does not delete original file.
    """

    os.system("cp " + filename + " " + output_name)

def get_output_filename(filename, output_num):
    """
    If filename starts with a number, replaces that num with output_num.
    Returns the new string.
    """

    filename = os.path.basename(filename)

    number_match = re.search(r'\d+', filename)
    if not number_match:
        return str(output_num) + filename

    return str(output_num) + filename[len(number_match.group()):]

def gen_colormaps():
    """
    Use not recommended - easier to use a screenshot with https://coolors.co/gradient-maker
    """
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

def callback_on_one(callback, filename, input_num, output_num, *args, **kwargs):
    """
    Executes callback(filename, output_name, kwargs) only if filename begins with input_num.
    output_name will be filename, but beginning with output_num instead of input_num.
    """

    search1 = re.search('[0-9]+', os.path.basename(filename))
    if search1 == None:
        return
    first_number = search1.group()

    match = first_number == str(input_num) and os.path.basename(filename).startswith(str(input_num))
    if not match:
        return

    output_name = get_output_filename(filename, output_num)
    callback(filename, output_name, *args, **kwargs)

def acceptable_extension(filename):
    """
    Returns whether filename has acceptable (can be manipulated) extension/file type.
    """
    
    return filename.endswith(
        (".png", ".gif", ".qt", ".mov", ".mp4")
    )    

def callback_all(callback, input_num, output_num, *args, **kwargs):
    """
    For all files in input_path starting with input_num, executes callback on them,
    saving callback's output to a filename beginning with output_num instead of input_num.

    Meant to be used as a wrapper around the other image manipulation functions.
    Since it lets you operate on image files in bulk.
    """
    
    print("\n--- Callback:", callback.__name__, "| Input num:", input_num, "| Output num:", output_num, "---")
    
    gather_files = [x for x in natsorted(os.listdir(input_path)) if acceptable_extension(x)]
    
    for filename in gather_files:
        full_filename = os.path.join(input_path, filename)
        callback_on_one(callback, full_filename, input_num, output_num, *args, **kwargs)

""" --- MAIN --- """

if __name__ == "__main__":

    # Example of using the image functions in bulk.
    # This will take a bunch of images starting with 0 and ending with .png and turn them into cool looking glitchy gifs (assuming you have the right libraries and command line functions working)
    # You can comment in/out specific parts of the pipeline if you want to swap out certain operations for other ones, or if you like the product of a certain step but not what happens after that.
    # Furthermore, since different input/output numbers are used for each step, you can check the results of each operation.

    callback_all(resize, 0, 1, max_height = 300)
    callback_all(pillow_adjust_image, 1, 2, new_contrast = 1.0, new_brightness = 1.1, new_saturation = 1.5)
    callback_all(apply_colormap, 2, 3, colormap_name = "colormap_greenhouse.png", gradient_alpha = 150)
    callback_all(glitch, 3, 6, glitch_amount = 2)
    callback_all(gif_reduce_frames, 6, 7, skip_frames = 6)
    callback_all(gif_change_speed, 7, 8, fps = 2)
    callback_all(gif_optimize, 8, 9, lossy=200, color_num = 16)

    callback_all(convert_gif, 2, 998, number_frames = 6, framerate = 3)
    callback_all(painting, 998, 999) 
    callback_all(gif_optimize, 999, 1000, lossy=200, color_num = 16)

    """
    # Example of using the image functions on 1 image.

    i = "0dorm.png"
    callback_on_one(pypxl, i, 0, 1)
    callback_on_one(glitch, i, 1, 1)
    """
