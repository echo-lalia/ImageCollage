
from PIL import Image, ImageChops, ImageFilter
import numpy as np
import argparse
import random
import os


DEFAULT_SCALE = 1.0
DEFAULT_COMPARE = 0.1
DEFAULT_LINEAR_WEIGHT = 1.0
DEFAULT_KERNEL_WEIGHT = 0.4
DEFAULT_OVERLAY = 0.0
DEFAULT_SUBTLE_OVERLAY = 0.3
DEFAULT_REPEAT_PENALTY = 0.2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ARG SETUP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# -----------------------------------------------------------------------------------------------------------------------------

def setup_args():
    global \
        show_preview, output_path, subtle_overlay_weight, overlay_weight, repeat_penalty, \
        kernel_error_weight, linear_error_weight, tile_directory, \
        tile_width, tile_height, compare_width, compare_height, \
        horizontal_tiles, vertical_tiles, source_image, \
        output_width, output_height, num_image_tiles


    parser = argparse.ArgumentParser(
        prog='hd_mosaic.py',
        description=ctext('This tool can be used to create a high-quality image mosaic by comparing given image tiles to a source image.', 'OKCYAN'),
        epilog=f"{ctext('Example:', 'HEADER')} {ctext('python3 hd_mosaic.py ../imagetiles ../targetimg.jpg 16x8', 'OKBLUE')}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    positional_group = parser.add_argument_group(title=ctext('positional', "HEADER"), description=ctext('These arguments are required.', 'OKCYAN'))
    scale_group = parser.add_argument_group(title=ctext('scale options', "HEADER"), description=ctext('These options take a float (which is multiplied by the original scale), or two integers separated by an "x".', 'OKCYAN'))
    weight_group = parser.add_argument_group(title=ctext('weight options', "HEADER"), description=ctext('These options take a float, and control the strength of different comparison strategies.', 'OKCYAN'))
    overlay_group = parser.add_argument_group(title=ctext('overlay options', "HEADER"), description=ctext('These options control the strength of the target image overlaid on top of the mosaic.', 'OKCYAN'))
    extra_group = parser.add_argument_group(title=ctext('additional options', "HEADER"), description=ctext('All other options.', 'OKCYAN'))
    # I'm just using this pattern to shrink the arg creation code
    for args, help, group, kwargs in [
        (('tile_folder',), 
            ("A path to a folder of images to build the mosaic with"), 
            positional_group,
            {'metavar':ctext(ctext('TILE_FOLDER', 'BOLD'), 'OKBLUE')},
        ),
        (('source_image',), 
            ("A path to an image to base the mosaic on."), 
            positional_group,
            {'metavar':ctext(ctext('SOURCE_IMAGE', 'BOLD'), 'OKBLUE')},
        ),
        (('tiles',), 
            ('The number of tiles to use. (EX: "16x10" sets 16 horizontal, 10 vertical tiles. "8" sets both to 8)'), 
            positional_group,
            {'metavar':ctext(ctext('TILES', 'BOLD'), 'OKBLUE')},
        ),
        (('-o','--output'), 
            ('The path (and/or filename) to use. Default is a generated filename in the current directory.'), 
            extra_group,
            {'metavar':ctext(ctext('PATH', 'OKBLUE'), 'OKBLUE')},
        ),
        (('-s','--scale'), 
            (f'The amount to rescale the input image.'), 
            scale_group,
            {'default':str(DEFAULT_SCALE), 'metavar':ctext('FLOAT|INTxINT', 'OKBLUE')},
        ),
        (('-c','--compare_scale'), 
            (f'The resolution that tiles will be compared at.'), 
            scale_group,
            {'default':str(DEFAULT_COMPARE), 'metavar':ctext('FLOAT|INTxINT', 'OKBLUE')},
        ),
        (('-l','--linear_error_weight'), 
            (f'How much the "linear" difference between pixels affects the output. '), 
            weight_group,
            {'type':float, 'default':DEFAULT_LINEAR_WEIGHT, 'metavar':ctext('FLOAT', 'OKBLUE')},
        ),
        (('-k','--kernel_error_weight'), 
            (f'How much the "kernel difference" comparison affects the output.'), 
            weight_group,
            {'type':float, 'default':DEFAULT_KERNEL_WEIGHT, 'metavar':ctext('FLOAT', 'OKBLUE')},
        ),
        (('-O','--overlay_opacity'), 
            (f'The alpha for a "normal" overlay of the target image over the mosaic.'), 
            overlay_group,
            {'type':float, 'default':DEFAULT_OVERLAY, 'metavar':ctext('FLOAT', 'OKBLUE')},
        ),
        (('-so','--subtle_overlay'), 
            (f'The alpha for an alternate, less sharp method of overlaying the target image on the mosaic.'), 
            overlay_group,
            {'type':float, 'default':DEFAULT_SUBTLE_OVERLAY, 'metavar':ctext('FLOAT', 'OKBLUE')},
        ),
        (('-r','--repeat_penalty'), 
            (f'How much to penalize repetition when selecting tiles.'), 
            weight_group,
            {'type':float, 'default':DEFAULT_REPEAT_PENALTY, 'metavar':ctext('FLOAT', 'OKBLUE')},
        ),
        (('-S','--show'), 
            ('If given, opens a preview of the output image upon completion.'), 
            extra_group,
            {'action':'store_true'},
        ),
        (('-h','--help'), 
            ('Print this help message.'), 
            extra_group,
            {'action':'help'},
        ),
    ]:
        if 'default' in kwargs:
            help += ctext(f"(default: {kwargs['default']})", 'OKBLUE')
        group.add_argument(*args, help=ctext(help, 'OKCYAN'), **kwargs)
    args = parser.parse_args()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Set up tile and source image resolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    

    # calculate real output size (for equally sized tiles)
    output_width = tile_width * horizontal_tiles
    output_height = tile_height * vertical_tiles

    tile_directory = args.tile_folder
    num_image_tiles = len(os.listdir(tile_directory))

    cprint('Converting source image...', 'OKBLUE')
    # resize input image for exact comparison with tiles
    og_width, og_height = source_image.width, source_image.height
    crop = crop_from_rescale((og_width, og_height), (output_width, output_height))
    source_image = source_image.resize((output_width, output_height), box=crop)
    # convert to Lab color space for more accurate comparisons
    source_image = source_image.convert(mode='LAB')


    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Finish setting up other input args ~~~~~~~~~~~~~~~~~~~~~~~~~
    # setup error weights
    linear_error_weight = args.linear_error_weight
    kernel_error_weight = args.kernel_error_weight

    # other image weights
    overlay_weight = args.overlay_opacity
    subtle_overlay_weight = args.subtle_overlay
    repeat_penalty = args.repeat_penalty


    # setup input/output paths
    output_path = args.output

    # create generated output filename
    if output_path is None \
    or output_path.endswith(os.path.sep):

        out_file = f'mosaic_{horizontal_tiles}x{vertical_tiles}'

        # add each non-default param to the filename
        for real_var, default_var, var_sym in [
            (compare_scale, DEFAULT_COMPARE, 'c'),
            (linear_error_weight, DEFAULT_LINEAR_WEIGHT, 'l'),
            (kernel_error_weight, DEFAULT_KERNEL_WEIGHT, 'k'),
            (repeat_penalty, DEFAULT_REPEAT_PENALTY, 'r'),
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

    show_preview = args.show

    # ~~~~~~~~~~~~~~~~~~~~~~~~~ Print friendly info about the current job ~~~~~~~~~~~~~~~~~~~~~~~~~
    cprint(f'Source image size: {og_width}x{og_height}', 'HEADER')
    cprint(f'{num_image_tiles} input tiles, {tile_width}x{tile_height}px each', 'HEADER')
    cprint(f'{horizontal_tiles}x{vertical_tiles} tiles in output image, totaling {horizontal_tiles * vertical_tiles}.', 'HEADER')
    cprint(f'Final output size: {output_width}x{output_height}', 'HEADER')
    cprint(f'Comparing at {compare_width}x{compare_height}px', 'HEADER')
    cprint(f'linear_weight: {linear_error_weight}, kernel_weight: {kernel_error_weight}, repeat_penalty: {repeat_penalty}', 'HEADER')
    cprint(f'Overlay alpha: {overlay_weight}, Subtle overlay alpha: {subtle_overlay_weight}', 'HEADER')


    mosaic = Mosaic(
        source_image=source_image,
        compare_size=Scale(compare_width, compare_height),
        tile_size=Scale(tile_width, tile_height),
        output_size=Scale(output_width, output_height),
        output_tiles_res=Scale(horizontal_tiles, vertical_tiles),
        linear_error_weight=linear_error_weight,
        kernel_error_weight=kernel_error_weight,
        overlay_alpha=overlay_weight,
        subtle_overlay_alpha=subtle_overlay_weight,
        tile_directory=tile_directory,
        repeat_penalty=repeat_penalty,
    )
    mosaic.fit_tiles()
    mosaic.save(output_path=output_path, show_preview=show_preview)
    cprint("Done!", 'OKGREEN')





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PRINT HELPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# -----------------------------------------------------------------------------------------------------------------------------
class Printer:
    """Simple helper for printing progress text."""
    max_line = ''
    load_chars = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]

    char_idx = 0

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

    def next_char(self):
        """Get the next loading character"""
        self.char_idx = (self.char_idx + 1) % len(self.load_chars)
        return self.load_chars[self.char_idx]
Printer = Printer()


def _pad_text(text, padding):
    return text + padding[len(text):]


def ctext(text, color) -> str:
    """Generate a colored string and return it."""
    if color.upper() in Printer.prntclrs:
        color = Printer.prntclrs[color.upper()]
    else:
        color = Printer.prntclrs['ENDC']
    return f"{color}{text}{Printer.prntclrs['ENDC']}"


def cprint(text, color):
    """Print in color (and pad lines to erase old text)"""
    text = str(text)
    if color.upper() in Printer.prntclrs:
        color = Printer.prntclrs[color.upper()]
    else:
        color = Printer.prntclrs['ENDC']
    text = _pad_text(text, Printer.max_line)
    print(f"{color}{text}{Printer.prntclrs['ENDC']}")


def cwrite(text):
    """Write to the terminal without starting a new line, erasing old text."""
    text = str(text)
    color = Printer.prntclrs['OKCYAN']
    text = f"{color}• {Printer.next_char()} - {text}{Printer.prntclrs['ENDC']}"
    if len(text) > len(Printer.max_line):
        Printer.max_line = ' ' * len(text)
    else:
        text = _pad_text(text, Printer.max_line)
    print(text, end='\r')




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Image functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# -----------------------------------------------------------------------------------------------------------------------------
def crop_from_rescale(old_size, new_size):
    ow, oh = old_size
    nw, nh = new_size
    width_shrink = (ow - nw) // 2
    height_shrink = (oh - nh) // 2
    return (
        width_shrink,
        height_shrink,
        ow - width_shrink,
        oh - height_shrink,
    )





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


def source_overlay(mosaic, source_image):
    overlay_img = ImageChops.overlay(mosaic, source_image)
    return ImageChops.blend(mosaic, overlay_img, overlay_weight)


def subtle_overlay(mosaic, source_image):
    """Soften edges on source image, and blur the image withing each tile, before applying the overlay."""
    source_image = source_image.convert(mode="RGB")
    mosaic = mosaic.convert(mode="RGB")

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

    source_image = ImageChops.overlay(mosaic, source_image)
    return ImageChops.blend(mosaic, source_image, subtle_overlay_weight)



class Scale:
    "A simple helper for bundling width/height information"
    def __init__(self, width, height):
        self.w = width
        self.h = height
    
    def __tuple__(self):
        return (self.w, self.h)

    def __iter__(self):
        for x in self.w, self.h:
            yield x
    
    def __len__(self):
        return 2
    
    def __getitem__(self, idx):
        if idx == 0:
            return self.w
        return self.h


class InputTile:
    tile_size = None
    compare_size = None
    kernel_error_weight = DEFAULT_KERNEL_WEIGHT
    linear_error_weight = DEFAULT_LINEAR_WEIGHT
    repeat_penalty = DEFAULT_REPEAT_PENALTY

    """A class to hold and work on input tiles"""
    def __init__(self, img):
        """Load one image tile, converted to proper output size"""
        if isinstance(img, Image.Image):
            self.img = img
        else:
            self.img = Image.open(img)

        # calculate image cropped size to match tile aspect ratio
        trgt_w, trgt_h = self._crop_from_ratio((self.img.width, self.img.height), self.tile_size)
        w_delta = self.img.width - trgt_w
        h_delta = self.img.height - trgt_h
        crop = (
            w_delta // 2,
            h_delta // 2,
            self.img.width - (w_delta // 2),
            self.img.height - (h_delta // 2),
        )
        # crop and resize image to tile size
        self.img = self.img.resize(self.tile_size, box=crop)

        # convert to Lab color space for more accurate comparisons
        try:
            self.img = self.img.convert(mode='LAB')
        except:
            # Sometimes the conversion fails due to the color mode the image loads with.
            self.img =  self.img.convert(mode='RGB').convert(mode="LAB")
        
        self.linear_array = self._as_linear_array()
        self.kernel_diff_array = self._as_kernel_diff_array()
        self.repeat_error = 0.0


    def add_to_mosaic(self, mosaic, crop):
        self.repeat_error += InputTile.repeat_penalty
        mosaic.paste(self.img, crop)


    def tile_error(self, source) -> float:
        """Get total error score for this tile"""
        err = self.repeat_penalty
        err += (np.mean(np.abs(source.linear_array - self.linear_array)) / 255) * InputTile.linear_error_weight
        err += (np.mean(np.abs(source.kernel_diff_array - self.kernel_diff_array)) / 255) * InputTile.kernel_error_weight
        return err


    def _as_kernel_diff_array(self):
        """
        For each pixel, avg val with neighbors to determine pixel kernel,
        return array representing the differences between the pixels and the pixel kernels.
        """
        img = self.img.resize((compare_width, compare_height))
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
        return arr


    def _as_linear_array(self):
        img = self.img.resize(self.compare_size)
        arr = np.array(
            [img.getpixel((x,y)) for y in range(img.height) for x in range(img.width)]
            )
        return arr


    def _crop_from_ratio(self, width_height, ratio):
        """Return the cropped width/height to match the given ratio"""
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
        
        return Scale(int(w), int(h))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mosaic ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# -----------------------------------------------------------------------------------------------------------------------------
class Mosaic:
    "A class to hold and work on the mosaic tiles."
    def __init__(
            self,
            source_image,
            tile_size,
            compare_size,
            output_size,
            output_tiles_res,
            linear_error_weight,
            kernel_error_weight,
            overlay_alpha,
            subtle_overlay_alpha,
            tile_directory,
            repeat_penalty,
    ):
        self.source_image = source_image
        InputTile.tile_size = tile_size
        InputTile.compare_size = compare_size
        self.output_size = output_size
        self.output_tiles_res = output_tiles_res


        self.tiles = self.load_tiles(tile_directory)

        InputTile.linear_error_weight = linear_error_weight
        InputTile.kernel_error_weight = kernel_error_weight
        InputTile.repeat_penalty = repeat_penalty

        self.overlay_alpha = overlay_alpha
        self.subtle_overlay_alpha = subtle_overlay_alpha

        # # pre-generate reusable np arrays to represent the tiles
        # if linear_error_weight:
        #     cprint("Generating arrays from tiles...", "OKBLUE")
        #     self.tile_arrays = [tile_to_array(tile) for tile in self.tiles]

        # # generate kernel diff arrays for each image
        # if kernel_error_weight:
        #     cprint("Generating kernel diff arrays...", "OKBLUE")
        #     self.kernel_diff_arrays = [tile_kernel_diff_array(tile) for tile in self.tiles]
        
        # store penalty for repeated tiles
        # self.repeat_penalties = np.array([0.0] * len(self.tiles))

        # finally, create the blank image to create our mosaic
        self.mosaic = Image.new(mode='LAB', size=tuple(self.output_size))


    def load_tiles(self, tile_directory):
        """Open, crop, and rescale tiles from tile directory"""
        tiles = []
        tile_idx = 0
        bad_tile_files = 0
        num_image_tiles = len(os.listdir(tile_directory))
        for img_file in os.scandir(tile_directory):
            tile_idx += 1
            cwrite(f'Loading tile {tile_idx}/{num_image_tiles} ({img_file.name})...')
            # PIL will determine what images are or are not valid.
            try:
                tiles.append(InputTile(img_file))
            except (OSError, ValueError):
                bad_tile_files += 1
                cprint(f"Warning: {img_file.name} could not be loaded", "WARNING")
        cprint(f"{num_image_tiles - bad_tile_files} tiles loaded.", "OKGREEN")
        return tiles


    def fit_tiles(self):
        """
        Scan through horizontal/vertical lines,
        matching a tile to each segment of the source image, 
        and add it to the mosaic.
        """
        horizontal_tiles, vertical_tiles = self.output_tiles_res
        tile_width, tile_height = InputTile.tile_size

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
                # crop region for comparison
                crop = (
                    tile_x * tile_width,
                    tile_y * tile_height,
                    tile_x * tile_width + tile_width,
                    tile_y * tile_height + tile_height,
                )
                # convert to an InputTile for comparison
                source_region = InputTile(self.source_image.crop(crop))

                # scan and find best matching tile image
                final_errors = [tile.compare(source_region) for tile in self.tiles]
                # final_errors = self.repeat_penalties

                # if self.linear_error_weight:
                #     final_errors = (
                #         final_errors
                #         + find_tile_errors(source_region, self.tile_arrays, tile_to_array)
                #         * self.linear_error_weight
                #     )

                # if self.kernel_error_weight:
                #     final_errors = (
                #         final_errors
                #         + find_tile_errors(source_region, self.kernel_diff_arrays, tile_kernel_diff_array)
                #         * self.kernel_error_weight
                #     )

                # get the first index where error was equal to the smallest error
                # select the tile at that index
                # best_idx, *_ = np.where(final_errors == final_errors.min())[0]
                best_idx = final_errors.index(min(final_errors))
                best_tile = self.tiles[best_idx]

                # track repeat penalty
                # self.repeat_penalties[best_idx] += repeat_penalty
                
                # add the best tile to the output image
                best_tile.add_to_mosaic(self.mosaic, crop)
                # self.mosaic.paste(best_tile, crop)

        cprint(f"{total_tiles} tiles selected.", "OKGREEN")

        if self.overlay_alpha:
            cprint('Applying overlay...', 'OKBLUE')
            self.mosaic = source_overlay(self.mosaic, self.source_image)
        
        if self.subtle_overlay_alpha:
            cprint('Applying subtle overlay...', 'OKBLUE')
            self.mosaic = subtle_overlay(self.mosaic, self.source_image)


    def save(self, show_preview, output_path):
        """Save (and/or show) the generated mosaic"""
        if show_preview:
            cprint('Showing img...', "OKBLUE")
            self.mosaic.show()

        cprint('Saving img...', "OKBLUE")
        self.mosaic = self.mosaic.convert(mode="RGB")
        self.mosaic.save(output_path)
        cprint(f'Saved as "{output_path}"', 'OKGREEN')



if __name__ == "__main__":
    setup_args()
