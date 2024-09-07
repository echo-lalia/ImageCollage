
from PIL import Image, ImageChops, ImageFilter
import numpy as np
import argparse
import random
import os


DEFAULT_SCALE = 1.0
DEFAULT_COMPARE = 0.1
DEFAULT_LINEAR_WEIGHT = 1.0
DEFAULT_KERNEL_WEIGHT = 0.4
DEFAULT_OVERLAY = 0.1
DEFAULT_SUBTLE_OVERLAY = 0.4
DEFAULT_REPEAT_PENALTY = 0.1



def setup_args():
    global show, output_path, subtle_overlay_weight, overlay_weight, repeat_penalty, \
        kernel_pixel_weight, linear_pixel_weight, compare_scale, tile_directory, \
        source_path, repeat_penalty, rescale_by, horizontal_tiles, vertical_tiles, \
        args, source_image, tile_width, tile_height, compare_width, compare_height


    parser = argparse.ArgumentParser(
        prog='ImageCollage',
        description='This tool can be used to create a high-quality collage by comparing given image tiles to a source image.'
    )
    # I'm just using this pattern to shrink the arg creation code
    for args, help, kwargs in [
        (('tile_folder',), 
            ("A path to a folder of images to build the mosaic with"), 
            {},
        ),
        (('source_image',), 
            ("A path to an image to base the mosaic on."), 
            {},
        ),
        (('tiles',), 
            ('The number of tiles (horizontal and/or vertical) to use.'), 
            {},
        ),
        (('-o','--output'), 
            ('The path (and/or filename) to use. Default is a generated filename in the current directory.'), 
            {},
        ),
        (('-s','--scale'), 
            (f'A float controlling the amount to rescale the input image. (Default {DEFAULT_SCALE})'), 
            {'default':str(DEFAULT_SCALE)},
        ),
        (('-c','--compare_scale'), 
            (f'The resolution scale that tiles will be compared at. (Default {DEFAULT_COMPARE})'), 
            {'default':str(DEFAULT_COMPARE)},
        ),
        (('-l','--linear_pixel_weight'), 
            (f'How much the "linear" difference between pixels affects the output. (Default {DEFAULT_LINEAR_WEIGHT})'), 
            {'type':float, 'default':DEFAULT_LINEAR_WEIGHT},
        ),
        (('-k','--kernel_pixel_weight'), 
            (f'How much the "kernel difference" comparison affects the output. (Default {DEFAULT_KERNEL_WEIGHT})'), 
            {'type':float, 'default':DEFAULT_KERNEL_WEIGHT},
        ),
        (('-O','--overlay_opacity'), 
            (f'If given, overlay original image on the collage using the given alpha. (Default {DEFAULT_OVERLAY})'), 
            {'type':float, 'default':DEFAULT_OVERLAY},
        ),
        (('-so','--subtle_overlay'), 
            (f'The alpha for an alternate, less sharp method of overlaying the target image. (Default {DEFAULT_SUBTLE_OVERLAY})'), 
            {'type':float, 'default':DEFAULT_SUBTLE_OVERLAY},
        ),
        (('-r','--repeat_penalty'), 
            (f'How much to penalize repetition when selecting tiles. (Default {DEFAULT_REPEAT_PENALTY})'), 
            {'type':float, 'default':DEFAULT_REPEAT_PENALTY},
        ),
        (('-p','--preview'), 
            ('If given, opens a preview of the output image upon completion.'), 
            {'action':'store_true'},
        ),
    ]:
        print(args, help, kwargs)
        parser.add_argument(*args, help=help, **kwargs)
    args = parser.parse_args()


    # parse tiles
    if 'x' in args.tiles.lower():
        ht, vt = args.tiles.lower().split('x')
        horizontal_tiles = int(ht)
        vertical_tiles = int(vt)
    else:
        horizontal_tiles = vertical_tiles = int(args.tiles)


    # friendly errors for wrong directories
    if os.path.isdir(args.source_image):
        raise ValueError('"source_image" is a directory.')
    if not os.path.isdir(args.tile_folder):
        raise ValueError('"tile_folder" should point to a folder full of images.')


    # parse image scale
    rescale_by = args.scale
    source_path = args.source_image
    source_image = Image.open(source_path)
    if "x" in rescale_by.lower():
        # width * height integers
        w, h = rescale_by.lower().split('x')
        source_image = source_image.resize((int(w), int(h)))
    else:
        # a single float
        rescale_by = float(rescale_by)
        if rescale_by != 1.0:
            source_image = source_image.resize((
                int(source_image.width * rescale_by),
                int(source_image.height * rescale_by),
            ))
    cprint('Successfully read source image', 'OKBLUE')


    # parse compare scale (and tile size)
    compare_scale = args.compare_scale
    tile_width = source_image.width // horizontal_tiles
    tile_height = source_image.height // vertical_tiles
    if 'x' in compare_scale.lower():
        # width * height integers
        w, h = compare_scale.lower().split('x')
        compare_width = int(w)
        compare_height = int(h)
    else:
        # a single float
        compare_scale = float(compare_scale)
        compare_width = int(tile_width * compare_scale)
        compare_height = int(tile_height * compare_scale)

    # setup error weights
    linear_pixel_weight = args.linear_pixel_weight
    kernel_pixel_weight = args.kernel_pixel_weight

    # other image weights
    overlay_weight = args.overlay_opacity
    subtle_overlay_weight = args.subtle_overlay
    repeat_penalty = args.repeat_penalty

    # setup input/output paths
    tile_directory = args.tile_folder
    output_path = args.output

    # create generated output filename
    if output_path is None \
    or output_path.endswith(os.path.sep):

        out_file = f'mosaic_{horizontal_tiles}x{vertical_tiles}'

        # add each non-default param to the filename
        for real_var, default_var, var_sym in [
            (compare_scale, DEFAULT_COMPARE, 'c'),
            (linear_pixel_weight, DEFAULT_LINEAR_WEIGHT, 'l'),
            (kernel_pixel_weight, DEFAULT_KERNEL_WEIGHT, 'k'),
            (repeat_penalty, DEFAULT_REPEAT_PENALTY, 'c'),
            (overlay_weight, DEFAULT_OVERLAY, 'O'),
            (subtle_overlay_weight, DEFAULT_SUBTLE_OVERLAY, 'so'),
        ]:
            if real_var != default_var:
                out_file += f"_{var_sym}{real_var}"

        out_file += '.jpg'
        if output_path is None:
            output_path = os.path.join(os.getcwd(), out_file)
        else:
            output_path = os.path.join(output_path, out_file)

    show = args.preview


prntclrs = {
    'HEADER':'\033[95m',
    'OKBLUE':'\033[94m',
    'OKCYAN':'\033[96m',
    'OKGREEN':'\033[92m',
    'WARNING':'\033[93m',
    'FAIL':'\033[91m',
    'ENDC':'\033[0m',
    'BOLD':'\033[1m',
    'UNDERLINE':'\033[4m',
}

class Printer:
    """Simple helper for printing progress text."""
    max_line = ''
    load_chars = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]

    char_idx = 0

    def next_char(self):
        """Get the next loading character"""
        self.char_idx = (self.char_idx + 1) % len(self.load_chars)
        return self.load_chars[self.char_idx]
Printer = Printer()


def _pad_text(text, padding):
    return text + padding[len(text):]


def cprint(text, color):
    """Print in color (and pad lines to erase old text)"""
    text = str(text)
    if color.upper() in prntclrs:
        color = prntclrs[color.upper()]
    else:
        color = prntclrs['ENDC']
    text = _pad_text(text, Printer.max_line)
    print(f"{color}{text}{prntclrs['ENDC']}")


def cwrite(text):
    """Write to the terminal without starting a new line, erasing old text."""
    text = str(text)
    color = prntclrs['OKCYAN']
    text = f"{color}• {Printer.next_char()} - {text}{prntclrs['ENDC']}"
    if len(text) > len(Printer.max_line):
        Printer.max_line = ' ' * len(text)
    else:
        text = _pad_text(text, Printer.max_line)
    print(text, end='\r')



def setup():
    global source_image, tile_width, tile_height, compare_width, compare_height
    global output_width, output_height, num_image_tiles



    # calculate real output size (for equally sized tiles)
    output_width = tile_width * horizontal_tiles
    output_height = tile_height * vertical_tiles

    num_image_tiles = len(os.listdir(tile_directory))

    cprint(f'Source image size: {source_image.width}x{source_image.height}', 'HEADER')
    cprint(f'{num_image_tiles} input tiles, {tile_width}x{tile_height}px each', 'HEADER')
    cprint(f'{horizontal_tiles}x{vertical_tiles} tiles in output image, totaling {horizontal_tiles * vertical_tiles}.', 'HEADER')
    cprint(f'Final output size: {output_width}x{output_height}', 'HEADER')
    cprint(f'Comparing at {compare_width}x{compare_height}px', 'HEADER')
    cprint(f'linear_weight: {linear_pixel_weight}, kernel_weight: {kernel_pixel_weight}, repeat_penalty: {repeat_penalty}', 'HEADER')
    cprint(f'Overlay alpha: {overlay_weight}, Subtle overlay alpha: {subtle_overlay_weight}', 'HEADER')

    cprint('Converting source image...', 'OKBLUE')
    # resize input image for easier comparison with tiles
    source_image = source_image.resize((output_width, output_height))

    # convert to Lab color space for more accurate comparisons
    source_image = source_image.convert(mode='LAB')


# def functions

def crop_from_ratio(width_height, ratio):
    w, h = width_height
    rw, rh = ratio

    s_width_factor = w / h
    r_width_factor = rw / rh

    width_factor = s_width_factor / r_width_factor
    
    # output must be smaller than input res
    if width_factor > 1:
        w /= width_factor
    else:
        h *= width_factor
    
    return int(w), int(h)


def load_image_tile(img_file):
    img = Image.open(img_file)

    # calculate image cropped size to match tile aspect ratio
    trgt_w, trgt_h = crop_from_ratio((img.width, img.height), (tile_width, tile_height))
    w_delta = img.width - trgt_w
    h_delta = img.height - trgt_h
    crop = (
        w_delta // 2,
        h_delta // 2,
        img.width - (w_delta // 2),
        img.height - (h_delta // 2),
    )
    # crop and resize image to tile size
    img = img.resize((tile_width, tile_height), box=crop)
    # convert to Lab color space for more accurate comparisons
    try:
        return img.convert(mode='LAB')
    except:
        # Sometimes the conversion fails due to the color mode the image loads with.
        return img.convert(mode='RGB').convert(mode="LAB")

def tile_error(source, tile) -> float:
    """Compare pixels in one tile, returning the error score"""

    return np.mean(np.abs(source - tile)) / 255


def find_tile_errors(source_region:Image.Image, tile_arrays:list, array_function:callable) -> np.array:
    """
    Convert source region using given array function, 
    and compare to tile arrays to find the error for each.
    array_function defaults to `tile_to_array`
    """
    
    source_arr = array_function(source_region)
    
    return np.array([tile_error(source_arr, tile_arr) for tile_arr in tile_arrays])


def tile_to_array(img):
    img = img.resize((compare_width, compare_height))
    arr = np.array(
        [img.getpixel((x,y)) for y in range(img.height) for x in range(img.width)]
        )
    return arr


def tile_kernel_diff_array(img):
    """
    For each pixel, avg val with neighbors to determine pixel kernel,
    return array representing the differences between the pixels and the pixel kernels.
    """
    img = img.resize((compare_width, compare_height))
    # represent image by nested arrays for simplicity
    arr = np.array([])
    for y in range(img.height):
        for x in range(img.width):
            # get all neighbors
            avg_val = 0
            for ky in range(3):
                for kx in range(3):
                    dx = x - 1 + kx
                    dy = y - 1 + ky
                    if 0 <= dx < img.width \
                    and 0 <= dy < img.height:
                        avg_val += img.getpixel((dx, dy))[0] / 9
            
            # compare to real val
            arr = np.append(arr, abs(img.getpixel((x,y))[0] - avg_val))
    return arr / 255


def source_overlay(collage, source_image):
    overlay_img = ImageChops.overlay(collage, source_image)
    return ImageChops.blend(collage, overlay_img, overlay_weight)


def subtle_overlay(collage, source_image):
    """Soften edges on source image, and blur the image withing each tile, before applying the overlay."""
    source_image = source_image.convert(mode="RGB")
    collage = collage.convert(mode="RGB")

    blur_amount = min(tile_width, tile_height) // 2

    # blur each tile segment of the source separately
    for tile_x in range(horizontal_tiles):
        for tile_y in range(vertical_tiles):

            crop = (
                tile_x * tile_width,
                tile_y * tile_height,
                tile_x * tile_width + tile_width,
                tile_y * tile_height + tile_height,
            )
            overlay_region = source_image.crop(crop)
            overlay_region = overlay_region.filter(ImageFilter.BoxBlur(blur_amount))
            source_image.paste(overlay_region, crop)

    source_image = ImageChops.overlay(collage, source_image)
    return ImageChops.blend(collage, source_image, subtle_overlay_weight)


# main
def main():
    global source_image, tile_width, tile_height, compare_width, compare_height
    global output_width, output_height, num_image_tiles

    tiles = []
    tile_idx = 0
    bad_tile_files = 0
    for img_file in os.scandir(tile_directory):
        tile_idx += 1
        cwrite(f'Loading tile {tile_idx}/{num_image_tiles} ({img_file.name})...')
        # PIL will determine what images are or are not valid.
        try:
            tiles.append(load_image_tile(img_file))
        except (OSError, ValueError):
            bad_tile_files += 1
            cprint(f"Warning: {img_file.name} could not be loaded", "WARNING")
    cprint(f"{num_image_tiles - bad_tile_files} tiles loaded.", "OKGREEN")
    
    # pre-generate reusable np arrays to represent the tiles
    if linear_pixel_weight:
        cprint("Generating arrays from tiles...", "OKBLUE")
        tile_arrays = [tile_to_array(tile) for tile in tiles]

    # generate kernel diff arrays for each image
    if kernel_pixel_weight:
        cprint("Generating kernel diff arrays...", "OKBLUE")
        kernel_diff_arrays = [tile_kernel_diff_array(tile) for tile in tiles]

    repeat_penalties = np.array([0.0] * len(tiles))
    
    collage = Image.new(mode='LAB', size=(output_width, output_height))

    # iterate over each tile
    # tile order is randomly shuffled so that repetition penalty doesn't favour top/left corner
    tile_ys = list(range(vertical_tiles))
    tile_xs = list(range(horizontal_tiles))
    random.shuffle(tile_ys)
    total_tiles = horizontal_tiles * vertical_tiles
    tile_idx = 0
    for tile_y in tile_ys:
        random.shuffle(tile_xs)
        for tile_x in tile_xs:

            tile_idx += 1
            cwrite(f"Comparing tile {tile_idx}/{total_tiles} ({tile_x}x{tile_y})...")

            # find the coordinate region of this tile
            crop = (
                tile_x * tile_width,
                tile_y * tile_height,
                tile_x * tile_width + tile_width,
                tile_y * tile_height + tile_height,
            )
            # crop region for comparison
            source_region = source_image.crop(crop)

            # scan and find best matching tile image

            final_errors = repeat_penalties

            if linear_pixel_weight:
                final_errors = (
                    final_errors
                    + find_tile_errors(source_region, tile_arrays, tile_to_array)
                    * linear_pixel_weight
                )

            if kernel_pixel_weight:
                final_errors = (
                    final_errors
                    + find_tile_errors(source_region, kernel_diff_arrays, tile_kernel_diff_array)
                    * kernel_pixel_weight
                )

            # get the first index where error was equal to the smallest error
            # select the tile at that index
            best_idx, *_ = np.where(final_errors == final_errors.min())[0]
            best_tile = tiles[best_idx]

            # track repeat penalty
            repeat_penalties[best_idx] += repeat_penalty
            
            # add the best tile to the output collage
            collage.paste(best_tile, crop)

    cprint(f"{total_tiles} tiles selected.", "OKGREEN")

    if overlay_weight:
        cprint('Applying overlay...', 'OKBLUE')
        collage = source_overlay(collage, source_image)

    if subtle_overlay_weight:
        cprint('Applying subtle overlay...', 'OKBLUE')
        collage = subtle_overlay(collage, source_image)

    if show:
        cprint('Showing img...', "OKBLUE")
        collage.show()

    cprint('Saving img...', "OKBLUE")
    collage = collage.convert(mode="RGB")
    collage.save(output_path)
    cprint(f'Saved as "{output_path}"', 'OKGREEN')
    cprint("Done!", 'OKGREEN')




if __name__ == "__main__":
    setup_args()
    setup()
    main()
