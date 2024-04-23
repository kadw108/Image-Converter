import os

import shlex

from PIL import Image, ImageOps
from dithering import ordered_dither
import numpy as np
import cv2

import math

from glitch_this import ImageGlitcher
glitcher = ImageGlitcher()

import re

current_path = os.path.abspath(os.getcwd())
input_path = current_path

""" IMAGE OR GIF ALTERING FUNCTIONS """

def posterize(filename, output_name):
    """
    Posterize input png.
    """

    inp = Image.open(filename)
    inp = ImageOps.posterize(inp, 3)
    inp.save(output_name)

def dither(filename, output_name, keep_transparency = True):
    """
    Dither input png.

    keep_transparency -- whether to preserve alpha in input png
    """

    inp = cv2.imread(filename)
    inp = ordered_dither(inp, "Bayer2x2")
    cv2.imwrite(output_name, inp)

    if keep_transparency:
        base_img = Image.open(filename).convert("RGBA")
        overlay_img = Image.open(output_name).convert("RGBA")
        alpha = base_img.split()[-1]
        mask_image = Image.new("RGBA", base_img.size, (0, 0, 0, 255))
        mask_image.paste(alpha, mask=alpha)
        mask_image = mask_image.convert('L')

        overlay_img.putalpha(mask_image)
        overlay_img.save(output_name)

        # real_output = Image.composite(base_img, overlay_img, mask_image)
        # real_output.save(output_name)

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

# pixelate
def pypxl(filename, output_name):
    """
    Apply pixelate effect to input png.
    """
    input_path2 = os.path.join(current_path, "pypxl")
    os.chdir(input_path2)

    command = "python3 pypxl_image.py -s 512 512 " + os.path.join(input_path, filename) + " " + os.path.join(input_path, output_name)
    os.system(command)

    os.chdir(current_path)

def apply_colormap(filename, output_name, colormap_name, gradient_alpha):
    """
    Applies gradient map/colormap to input png.

    colormap_name -- path to colormap, 1d horizontal pixel strip with lighter colors to the right
    gradient_alpha -- strength of colormap; number from [0, 255] with higher = stronger colormap
    """

    # from https://stackoverflow.com/a/71584672
    im = Image.open(filename).convert('RGB')
    na = np.array(im)
    grey = np.mean(na, axis=2).astype(np.uint8)

    # Load colourmap
    cmap = Image.open(colormap_name).convert('RGB')

    # Make output image, same height and width as grey image, but 3-channel RGB
    result = np.zeros((*grey.shape,3), dtype=np.uint8)

    # Take entries from RGB colourmap according to greyscale values in image
    np.take(cmap.getdata(), grey, axis=0, out=result)

    # overlay_img has map applied in full
    overlay_img = Image.fromarray(result).convert("RGB")
    overlay_img.save(output_name)

    base_img = Image.open(filename).convert("RGB")
    overlay_img.putalpha(gradient_alpha)
    base_img.paste(overlay_img, mask=overlay_img)
    base_img.save(output_name)

def glitch(filename, output_name):
    """
    Turns input png into glitchy gif.
    """
    # glitch_image(self, src_img, glitch_amount, glitch_change=0.0, cycle=False, color_offset=False, scan_lines=False, gif=False, frames=23, step=1)
    glitch_img = glitcher.glitch_image(filename, 0.7, color_offset = True, gif=True)
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

""" UTILITY FUNCTIONS """

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
    if not os.path.basename(filename).startswith(str(input_num)):
        return
    output_name = get_output_filename(filename, output_num)
    callback(filename, output_name, *args, **kwargs)

def callback_all(callback, input_num, output_num, *args, **kwargs):
    """
    For all files in input_path starting with input_num, executes callback on them,
    saving callback's output to a filename beginning with output_num instead of input_num.

    Meant to be used as a wrapper around the other image manipulation functions.
    Since it lets you operate on image files in bulk.
    """
    print("\n--- Callback:", callback.__name__, "| Input num:", input_num, "| Output num:", output_num, "---")
    gather_files = [x for x in sorted(os.listdir(input_path)) if (x.endswith(".png") or x.endswith(".gif"))]
    for filename in gather_files:
        full_filename = os.path.join(input_path, filename)
        callback_on_one(callback, full_filename, input_num, output_num, *args, **kwargs)

""" MAIN """

if __name__ == "__main__":

    # Example of using the image functions in bulk.
    # This will take a bunch of images starting with 0 and ending with .png and turn them into cool looking glitchy gifs (assuming you have the right libraries and command line functions working)
    # You can comment in/out specific parts of the pipeline if you want to swap out certain operations for other ones, or if you like the product of a certain step but not what happens after that.
    # Furthermore, since different input/output numbers are used for each step, you can check the results of each operation.

    callback_all(resize, 0, 2, max_height = 450)
    callback_all(apply_colormap, 2, 3, colormap_name = "colormap_greenhouse.png", gradient_alpha = 150)
    callback_all(glitch, 3, 4)
    callback_all(gif_change_speed, 4, 5, fps = 6)
    callback_all(gif_downsize, 5, 6, percent = 80)
    callback_all(gif_reduce_frames, 5, 7)
    callback_all(gif_change_speed, 7, 8, fps = 6)
    callback_all(gif_optimize, 8, 9, lossy=200, color_num = 64)

    """
    # Example of using the image functions on 1 image.

    i = "0dorm.png"
    callback_on_one(pypxl, i, 0, 1)
    callback_on_one(glitch, i, 1, 1)
    """
