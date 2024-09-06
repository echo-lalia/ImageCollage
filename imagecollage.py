
from PIL import Image, ImageChops
import numpy as np
import argparse
import random
import os


DEFAULT_SCALE = 1.0
DEFAULT_COMPARE = 0.1
DEFAULT_LINEAR_WEIGHT = 1.0
DEFAULT_KERNEL_WEIGHT = 0.3
DEFAULT_ENTROPY_WEIGHT = 0.0
DEFAULT_OVERLAY = 0.1
DEFAULT_REPEAT_PENALTY = 0.1



parser = argparse.ArgumentParser(
    prog='ImageCollage',
    description='This tool can be used to create a high-quality collage by comparing given image tiles to a source image.'
)
parser.add_argument(
    'source_image',
    help="A path to an image to base the collage on."
    )
parser.add_argument(
    'tile_folder',
    help="A path to a folder of images to build the collage with."
    )
parser.add_argument(
    '-o', '--output',
    help='The path (and/or filename) to use. Default is a generated filename in the current directory.'
    )
parser.add_argument(
    '-t', '--tiles', type=int,
    help='The number of tiles (horizontal and vertical) to use.'
    )
parser.add_argument(
    '-z', '--horizontal_tiles', type=int,
    help='The number of horizontal tiles to use.'
    )
parser.add_argument(
    '-v', '--vertical_tiles', type=int,
    help='The number of vertical tiles to use.'
    )
parser.add_argument(
    '-s', '--scale', 
    default=DEFAULT_SCALE, type=float,
    help=f'A float controlling the amount to rescale the input image. (Default {DEFAULT_SCALE})'
    )
parser.add_argument(
    '-c', '--compare_scale', 
    default=DEFAULT_COMPARE, type=float,
    help=f'The resolution scale that tiles will be compared at. (Default {DEFAULT_COMPARE})'
    )
parser.add_argument(
    '-l', '--linear_pixel_weight', 
    default=DEFAULT_LINEAR_WEIGHT, type=float,
    help=f'How much the "linear" difference between pixels affects the output. (Default {DEFAULT_LINEAR_WEIGHT})'
    )
parser.add_argument(
    '-k', '--kernel_pixel_weight', 
    default=DEFAULT_KERNEL_WEIGHT, type=float,
    help=f'How much the "kernel difference" comparison affects the output. (Default {DEFAULT_KERNEL_WEIGHT})'
    )
parser.add_argument(
    '-e', '--entropy_weight', 
    default=DEFAULT_ENTROPY_WEIGHT, type=float,
    help=f'How much the "entropy" between tiles affects the output. (Default {DEFAULT_ENTROPY_WEIGHT})'
    )
parser.add_argument(
    '-O', '--overlay_opacity', 
    default=DEFAULT_OVERLAY, type=float,
    help=f'If given, overlay original image on the collage using the given opacity. (Default {DEFAULT_OVERLAY})'
    )
parser.add_argument(
    '-r', '--repeat_penalty',
    default=DEFAULT_REPEAT_PENALTY, type=float,
    help=f'How much to penalize repetition when selecting tiles. (Default {DEFAULT_REPEAT_PENALTY})'
    )
parser.add_argument(
    '-p', '--preview',
    action='store_true',
    help='If given, opens a preview of the output image upon completion.'
    )
args = parser.parse_args()


# setup input vars
if args.tiles is None and args.horizontal_tiles is None:
    raise ValueError('"tiles" or "horizontal_tiles" param must be given.')
if args.tiles is None and args.vertical_tiles is None:
    raise ValueError('"tiles" or "vertical_tiles" param must be given.')

horizontal_tiles = args.tiles if args.horizontal_tiles is None else args.horizontal_tiles
vertical_tiles = args.tiles if args.vertical_tiles is None else args.vertical_tiles

# how much to scale input image by BEFORE processing
rescale_by = args.scale
compare_scale = args.compare_scale

linear_pixel_weight = args.linear_pixel_weight
kernel_pixel_weight = args.kernel_pixel_weight
entropy_weight = args.entropy_weight

overlay_weight = args.overlay_opacity
repeat_penalty = args.repeat_penalty

source_path = args.source_image
tile_directory = args.tile_folder

output_path = args.output
if output_path is None \
or output_path.endswith(os.path.sep):
    # assemble a filename
    out_file = f'collage_{horizontal_tiles}x{vertical_tiles}_c{compare_scale}'
    if linear_pixel_weight:
        out_file += f'_l{linear_pixel_weight}'
    if kernel_pixel_weight:
        out_file += f'_k{kernel_pixel_weight}'
    if entropy_weight:
        out_file += f'_e{entropy_weight}'
    if repeat_penalty:
        out_file += f'_r{repeat_penalty}'
    if overlay_weight:
        out_file += f'_o{overlay_weight}'
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
    max_line = ''
    # load_chars = ['|', '/', '-', '\\']
    load_chars = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]

    char_idx = 0

    def next_char(self):
        self.char_idx = (self.char_idx + 1) % len(self.load_chars)
        return self.load_chars[self.char_idx]

Printer = Printer()


def cprint(text, color):
    text = str(text)
    if color.upper() in prntclrs:
        color = prntclrs[color.upper()]
    else:
        color = prntclrs['ENDC']
    print(Printer.max_line, end='\r')
    print(f"{color}{text}{prntclrs['ENDC']}")


def cwrite(text):
    text = str(text)
    color = prntclrs['OKCYAN']
    text = f"{color}• {Printer.next_char()} - {text}{prntclrs['ENDC']}"
    if len(text) > len(Printer.max_line):
        Printer.max_line = ' ' * len(text)
    print(Printer.max_line, end='\r')
    print(text, end='\r')



def setup():
    global source_image, tile_width, tile_height, compare_width, compare_height
    global output_width, output_height, num_image_tiles

    # read input image and calculate values for the script
    source_image = Image.open(source_path)
    source_image = source_image.resize((
        int(source_image.width * rescale_by),
        int(source_image.height * rescale_by),
    ))

    cprint('Successfully read source image', 'OKBLUE')


    tile_width = source_image.width // horizontal_tiles
    tile_height = source_image.height // vertical_tiles
    compare_width = int(tile_width * compare_scale)
    compare_height = int(tile_height * compare_scale)


    # calculate real output size (for equally sized tiles)
    output_width = tile_width * horizontal_tiles
    output_height = tile_height * vertical_tiles

    num_image_tiles = len(os.listdir(tile_directory))

    cprint(f'Source image size: {source_image.width}x{source_image.height}', 'HEADER')
    cprint(f'{num_image_tiles} input tiles, {tile_width}x{tile_height}px each', 'HEADER')
    cprint(f'{horizontal_tiles}x{vertical_tiles} tiles in output image, totaling {horizontal_tiles * vertical_tiles}.', 'HEADER')
    cprint(f'Comparing at {compare_width}x{compare_height}px', 'HEADER')
    cprint(f'Final output size: {output_width}x{output_height}', 'HEADER')


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
    return img.convert(mode='LAB')


def arrays_from_tiles(tiles):
    """Generate np arrays representing the given tiles"""
    cprint("Generating arrays from tiles...", "OKBLUE")
    tile_arrays = []
    for tile in tiles:
        tile_arrays.append(tile_to_array(tile))
    return tile_arrays


def tile_error(source, tile) -> float:
    """Compare pixels in one tile, returning the error score"""

    return np.mean(np.abs(source - tile)) / 255


def find_linear_best_tile(source_region:Image.Image, tiles:list, tile_arrays:list) -> Image.Image:
    """Compare pixels in each tile to find closest match"""
    # generate array to represent source region
    source_arr = tile_to_array(source_region)

    errors = np.array([tile_error(source_arr, tile_arr) for tile_arr in tile_arrays])
    return errors


def find_kernel_best_tile(source_region:Image.Image, tiles:list, kernel_diff_arrays:list) -> Image.Image:
    """Compare pixels in each tile to find closest match"""
    # generate array to represent source region
    source_arr = tile_kernel_diff_array(source_region)

    errors = np.array([tile_error(source_arr, tile_arr) for tile_arr in kernel_diff_arrays])
    return errors



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


def kernel_diff_from_tiles(tiles):
    """Generate np arrays representing the local entropy of each pixel"""
    cprint("Generating kernel diff arrays...", "OKBLUE")
    tile_arrays = []
    for tile in tiles:
        tile_arrays.append(tile_kernel_diff_array(tile))
    return tile_arrays


def source_overlay(collage, source_image):
    overlay_img = ImageChops.overlay(collage, source_image)
    return ImageChops.blend(collage, overlay_img, overlay_weight)


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
    tile_arrays = arrays_from_tiles(tiles)

    # generate entropy arrays for each image
    kernel_diff_arrays = kernel_diff_from_tiles(tiles)

    # generate entropies for each image
    entropies = [tile.entropy() for tile in tiles]

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
            linear_errors = find_linear_best_tile(source_region, tiles, tile_arrays)
            kernel_errors = find_kernel_best_tile(source_region, tiles, kernel_diff_arrays)

            # compare entropy of source to tiles
            source_entropy = source_region.entropy()
            entropy_errors = np.array([abs(source_entropy - entropy) for entropy in entropies])
            # scale by max entropy to make value more predictable
            if max(np.abs(entropies)) != 0:
                entropy_errors /= max(np.abs(entropies))

            final_errors = (linear_errors * linear_pixel_weight) \
                         + (kernel_errors * kernel_pixel_weight) \
                         + (entropy_errors * entropy_weight) \
                         + repeat_penalties

            # get the first index where error was equal to the smallest error
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

    if show:
        cprint('Showing img...', "OKBLUE")
        collage.show()

    cprint('Saving img...', "OKBLUE")
    collage = collage.convert(mode="RGB")
    collage.save(output_path)
    cprint(f'Saved as "{output_path}"', 'OKGREEN')
    cprint("Done!", 'OKGREEN')


    





if __name__ == "__main__":
    setup()
    main()
