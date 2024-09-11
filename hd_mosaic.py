
from PIL import Image, ImageOps, ImageChops, ImageFilter, ImageEnhance
import numpy as np
import argparse
import random
import shutil
import time
import sys
import os


DEFAULT_SCALE = 1.0
DEFAULT_COMPARE = 0.1
MAX_DEFAULT_COMPARE_RES = 9
DEFAULT_LINEAR_WEIGHT = 1.0
DEFAULT_KERNEL_WEIGHT = 0.1
DEFAULT_OVERLAY = 0.0
DEFAULT_SUBTLE_OVERLAY = 0.5
DEFAULT_REPEAT_PENALTY = 0.1
DEFAULT_SUBDIVISIONS = 1
DEFAULT_SUBDIVISION_THRESHOLD = 150

VERBOSE = False
USER_INPUT = None




class InputParameter:
    """This class stores information about an input parameter"""
    def __init__(
            self,
            name,
            help="",
            type=str,
            prompt=None,
            metavar=None,
            default=None,
            static=False,
            ):
        self.name = name
        self.help = help
        self.prompt = f"{name}: {help}" if prompt is None else prompt
        self.type = type
        self.metavar = metavar
        self.default = default
        self.value = default
        self.static = static
    
    def prompt_string(self, add_in=None):
        string = f"\n\n{ctext(ctext(self.name, "HEADER"), "BOLD")} : {ctext(self.metavar, "GRAY")}"
        if self.value is not None:
            string += f" = {ctext(str(self.value), 'GRAY')}"
        if self.default is not None:
            string += ctext(f" (default: {self.default})", "GRAY")
        string += ctext(f"\n{self.help}", 'gray')
        string += ctext(f"\n\n{self.prompt}", "OKBLUE")
        if add_in is not None:
            string += f"\n\n{add_in}"
        string += ctext(f"\n\n{self.name} = ", "HEADER")
        return string


class InputAction:
    def __init__(
            self,
            name,
            description,
            callback,
    ):
        self.name = name
        self.description = description
        self.callback = callback



class UserInput:
    """This class is designed to get input variables from the user, using both argparse and input."""
    def __init__(self, name):
        self.name = name
        self.categories = {"":{"desc":"", "params":[]}}
        self.params = {}
        self.actions = {}

    def __getitem__(self, key):
        return self.params[key]

    def add_category(self, category, description):
        self.categories[category] = {"desc":description, "params":[]}

    def add_parameter(
            self,
            name,
            category="",
            **kwargs,
    ):
        param = InputParameter(name, **kwargs)
        self.categories[category]["params"].append(name)
        self.params[name] = param
    
    def add_action(
            self,
            name,
            desc,
            callback,
    ):
        self.actions.append(
            InputAction(name, desc, callback)
        )
        

    def get_param(self, name, prev_message=""):
        param = self[name]

        add_in = None

        # get user input for this param, and set it
        # if there is an error, print it nicely and keep trying:
        while True:
            try:
                # clear screen
                size = shutil.get_terminal_size()
                for _ in range(size.lines):
                    print()

                print(prev_message)

                inpt = input(param.prompt_string(add_in=add_in))
                if inpt == "exit":
                    sys.exit()
                param.value = param.type(inpt)
                return ctext(f"Set {param.name} to {param.value}.", 'OKGREEN')
            except (ValueError) as e:
                add_in = ctext(getattr(e, 'message', str(e)), 'WARNING')

    def get_static(self):
        """Fill the static parameters"""
        prev_message = ""
        for name, param in self.params.items():
            if param.static:
                prev_message = self.get_param(name, prev_message=prev_message)
        if prev_message:
            print(prev_message)


    def main_menu(self):
        """Show the main menu"""



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ARG SETUP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# -----------------------------------------------------------------------------------------------------------------------------
def main():
    global VERBOSE, USER_INPUT
    USER_INPUT = ui = UserInput("hd_mosaic")

    ui.add_category("input/output", "Options for the input/output of images")
    ui.add_category("tiles", "Options relating to the tiles")
    ui.add_category("weights", "Weights to use for different comparison methods")
    ui.add_category("tiles", "Options relating to the tiles")
    ui.add_category("overlay", "Options relating to the image overlay")
    ui.add_category("subdivision", "Options relating to the tile subdivision")
    ui.add_category("other", "Uncategorized options")


    ui.add_parameter(
        "tile_folder", category="input/output", 
        help="The source folder containing all the image tiles.",
        prompt="Please provide the path to a folder of images to use as tiles:",
        metavar="PATH", static=True, 
        )
    ui.add_parameter(
        "tile_resolution", category="tiles", 
        help="The resolution to load the tiles in with.", 
        prompt="""Please provide the resolution you would like to load the tiles in with: 
(Really small resolutions will look bad, really big resolutions will be very slow)""",
        metavar="INT|INTxINT", type=InputScale, static=True,
        )
    ui.add_parameter(
        "source_image", category="input/output", 
        help="The source image to base the mosaic on.", 
        prompt="Provide the path to an image to base this mosaic on:",
        metavar="PATH",
        )
    ui.add_parameter(
        "output", category="input/output", 
        help="The source image to base the mosaic on.", 
        prompt="Provide the path to an image to base this mosaic on:",
        metavar="PATH",
        )
    

    ui.get_static()

    


# def main():
#     global \
#         show_preview, output_path, subtle_overlay_weight, overlay_weight, repeat_penalty, \
#         kernel_error_weight, linear_error_weight, tile_directory, \
#         tile_width, tile_height, compare_width, compare_height, \
#         horizontal_tiles, vertical_tiles, source_image, \
#         output_width, output_height, num_image_tiles, VERBOSE


#     parser = argparse.ArgumentParser(
#         description=ctext('This tool can be used to create a high-quality image mosaic by comparing given image tiles to a source image.', 'DARKBLUE'),
#         epilog=f"{ctext('Example:', 'DARKBLUE')} {ctext('python3 hd_mosaic.py ../imagetiles ../targetimg.jpg 16x8', 'GRAY')}",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         add_help=False,
#     )
#     positional_group = parser.add_argument_group(title=ctext('positional', "OKBLUE"), description=ctext('These arguments are required.', 'DARKBLUE'))
#     scale_group = parser.add_argument_group(title=ctext('scale options', "OKBLUE"), description=ctext('These options take a float (which is multiplied by the original scale), or two integers separated by an "x".', 'DARKBLUE'))
#     weight_group = parser.add_argument_group(title=ctext('weight options', "OKBLUE"), description=ctext('These options take a float, and control the strength of different comparison strategies.', 'DARKBLUE'))
#     overlay_group = parser.add_argument_group(title=ctext('overlay options', "OKBLUE"), description=ctext('These options control the strength of the target image overlaid on top of the mosaic.', 'DARKBLUE'))
#     extra_group = parser.add_argument_group(title=ctext('additional options', "OKBLUE"), description=ctext('All other options.', 'DARKBLUE'))
#     # I'm just using this pattern to shrink the arg creation code
#     for args, help, group, kwargs in [
#         (('tile_folder',), 
#             ("A path to a folder of images to build the mosaic with"), 
#             positional_group,
#             {'metavar':'TILE_FOLDER'},
#         ),
#         (('source_image',), 
#             ("A path to an image to base the mosaic on."), 
#             positional_group,
#             {'metavar':'SOURCE_IMAGE'},
#         ),
#         (('tiles',), 
#             ('The number of tiles to use. (EX: "16x10" sets 16 horizontal, 10 vertical tiles. "8" sets both to 8)'), 
#             positional_group,
#             {'metavar':'TILES'},
#         ),
#         (('-o','--output'), 
#             ('The path (and/or filename) to use. Default is a generated filename in the current directory.'), 
#             extra_group,
#             {'metavar':'PATH'},
#         ),
#         (('-s','--scale'), 
#             (f'The amount to rescale the input image.'), 
#             scale_group,
#             {'default':str(DEFAULT_SCALE), 'metavar':'FLOAT|INTxINT'},
#         ),
#         (('-c','--compare_scale'), 
#             (f'The resolution that tiles will be compared at.'), 
#             scale_group,
#             {'default':str(DEFAULT_COMPARE), 'metavar':'FLOAT|INTxINT'},
#         ),
#         (('-l','--linear_error_weight'), 
#             (f'How much the "linear" difference between pixels affects the output. '), 
#             weight_group,
#             {'type':float, 'default':DEFAULT_LINEAR_WEIGHT, 'metavar':'FLOAT'},
#         ),
#         (('-k','--kernel_error_weight'), 
#             (f'How much the "kernel difference" comparison affects the output.'), 
#             weight_group,
#             {'type':float, 'default':DEFAULT_KERNEL_WEIGHT, 'metavar':'FLOAT'},
#         ),
#         (('-O','--overlay_opacity'), 
#             (f'The alpha for a "normal" overlay of the target image over the mosaic.'), 
#             overlay_group,
#             {'type':float, 'default':DEFAULT_OVERLAY, 'metavar':'FLOAT'},
#         ),
#         (('-so','--subtle_overlay'), 
#             (f'The alpha for an alternate, less sharp method of overlaying the target image on the mosaic.'), 
#             overlay_group,
#             {'type':float, 'default':DEFAULT_SUBTLE_OVERLAY, 'metavar':'FLOAT'},
#         ),
#         (('-r','--repeat_penalty'), 
#             (f'How much to penalize repetition when selecting tiles.'), 
#             weight_group,
#             {'type':float, 'default':DEFAULT_REPEAT_PENALTY, 'metavar':'FLOAT'},
#         ),
#         (('-S','--show'), 
#             ('If given, opens a preview of the output image upon completion.'), 
#             extra_group,
#             {'action':'store_true'},
#         ),
#         (('-d','--subdivisions'), 
#             ('Max number of subdivisions allowed in each main tile.'), 
#             extra_group,
#             {'type':int, 'default':DEFAULT_SUBDIVISIONS, 'metavar':'INT'},
#         ),
#         (('-D','--detail_map'), 
#             ('An image that controls where extra subdivisions are added.'), 
#             extra_group,
#             {'metavar':'PATH'},
#         ),
#         (('-t','--subdivision_threshold'), 
#             ('Detail values higher than this threshold will create a subdivision.'), 
#             extra_group,
#             {'type':int, 'default':DEFAULT_SUBDIVISION_THRESHOLD, 'metavar':'INT'},
#         ),
#         (('-V','--verbose'), 
#             ('Print additional debug information.'), 
#             extra_group,
#             {'action':'store_true'},
#         ),
#         (('-h','--help'), 
#             ('Print this help message.'), 
#             extra_group,
#             {'action':'help'},
#         ),
#     ]:
#         if 'metavar' in kwargs:
#             if group is positional_group:
#                 kwargs['metavar'] = ctext(kwargs['metavar'], 'BOLD')
#             else:
#                 kwargs['metavar'] = ctext(kwargs['metavar'], 'GRAY')
#         if 'default' in kwargs:
#             help += ctext(f"(default: {kwargs['default']})", 'GRAY')
#         group.add_argument(*args, help=ctext(help, 'DARKBLUE'), **kwargs)
#     args = parser.parse_args()
#     VERBOSE = args.verbose

#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Set up tile and source image resolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     # parse tiles
#     if 'x' in args.tiles.lower():
#         ht, vt = args.tiles.lower().split('x')
#         horizontal_tiles = int(ht)
#         vertical_tiles = int(vt)
#     else:
#         horizontal_tiles = vertical_tiles = int(args.tiles)


#     # friendly errors for wrong directories
#     if os.path.isdir(args.source_image):
#         raise ValueError('"source_image" is a directory.')
#     if not os.path.isdir(args.tile_folder):
#         raise ValueError('"tile_folder" should point to a folder full of images.')

#     # parse image scale
#     rescale_by = args.scale
#     source_path = args.source_image
#     source_image = Image.open(source_path)
#     if "x" in rescale_by.lower():
#         # width * height integers
#         w, h = rescale_by.lower().split('x')
#         source_image = source_image.resize((int(w), int(h)))
#     else:
#         # a single float
#         rescale_by = float(rescale_by)
#         if rescale_by != 1.0:
#             source_image = source_image.resize((
#                 int(source_image.width * rescale_by),
#                 int(source_image.height * rescale_by),
#             ))
#     cprint('Successfully read source image', 'OKBLUE')

#     # parse compare scale (and tile size)
#     compare_scale = args.compare_scale
#     # tile size must divide evenly into subdivision width
#     subdivisions = args.subdivisions
#     sub_width = (2**subdivisions)
#     tile_width = (source_image.width // (horizontal_tiles * sub_width)) * sub_width
#     tile_height = (source_image.height // (vertical_tiles * sub_width)) * sub_width
#     if 'x' in compare_scale.lower():
#         # width * height integers
#         w, h = compare_scale.lower().split('x')
#         compare_width = int(w)
#         compare_height = int(h)
#     else:
#         # a single float
#         compare_scale = float(compare_scale)
#         compare_width = int(tile_width * compare_scale)
#         compare_height = int(tile_height * compare_scale)
#         # clamp default compare to a reasonable value
#         if compare_scale == DEFAULT_COMPARE:
#             compare_width = min(max(1, compare_width), MAX_DEFAULT_COMPARE_RES)
#             compare_height = min(max(1, compare_height), MAX_DEFAULT_COMPARE_RES)
    

#     # calculate real output size (for equally sized tiles)
#     output_width = tile_width * horizontal_tiles
#     output_height = tile_height * vertical_tiles

#     tile_directory = args.tile_folder
#     num_image_tiles = len(os.listdir(tile_directory))

#     cprint('Converting source image...', 'OKBLUE')
#     # resize input image for exact comparison with tiles
#     og_width, og_height = source_image.width, source_image.height
#     crop = crop_from_rescale((og_width, og_height), (output_width, output_height))
#     source_image = source_image.resize((output_width, output_height), box=crop)
#     # convert to Lab color space for more accurate comparisons
#     source_image = source_image.convert(mode='RGB')


#     # ~~~~~~~~~~~~~~~~~~~~~~~~~ Finish setting up other input args ~~~~~~~~~~~~~~~~~~~~~~~~~
#     # setup error weights
#     linear_error_weight = args.linear_error_weight
#     kernel_error_weight = args.kernel_error_weight

#     # other image weights
#     overlay_weight = args.overlay_opacity
#     subtle_overlay_weight = args.subtle_overlay
#     repeat_penalty = args.repeat_penalty


#     # setup input/output paths
#     output_path = args.output

#     # create generated output filename
#     if output_path is None \
#     or output_path.endswith(os.path.sep):

#         out_file = f'mosaic_{horizontal_tiles}x{vertical_tiles}'

#         # add each non-default param to the filename
#         for real_var, default_var, var_sym in [
#             (compare_scale, DEFAULT_COMPARE, 'c'),
#             (linear_error_weight, DEFAULT_LINEAR_WEIGHT, 'l'),
#             (kernel_error_weight, DEFAULT_KERNEL_WEIGHT, 'k'),
#             (repeat_penalty, DEFAULT_REPEAT_PENALTY, 'r'),
#             (overlay_weight, DEFAULT_OVERLAY, 'O'),
#             (subtle_overlay_weight, DEFAULT_SUBTLE_OVERLAY, 'so'),
#         ]:
#             if real_var != default_var:
#                 out_file += f"_{var_sym}{real_var}"

#         out_file += '.jpg'
#         if output_path is None:
#             output_path = os.path.join(os.getcwd(), out_file)
#         else:
#             output_path = os.path.join(output_path, out_file)

#     show_preview = args.show

#     if args.detail_map:
#         detail_map = Image.open(args.detail_map)
#     else:
#         detail_map = None

#     # ~~~~~~~~~~~~~~~~~~~~~~~~~ Print friendly info about the current job ~~~~~~~~~~~~~~~~~~~~~~~~~
#     cprint(f'Source image size: {og_width}x{og_height}', 'HEADER')
#     cprint(f'{num_image_tiles} input tiles, {tile_width}x{tile_height}px each', 'HEADER')
#     cprint(f'{horizontal_tiles}x{vertical_tiles} tiles in output image, totaling {horizontal_tiles * vertical_tiles} (up to {horizontal_tiles*sub_width * vertical_tiles*sub_width} with subdivisions).', 'HEADER')
#     cprint(f'Final output size: {output_width}x{output_height}', 'HEADER')
#     cprint(f'Comparing at {compare_width}x{compare_height}px', 'HEADER')
#     cprint(f'linear_weight: {linear_error_weight}, kernel_weight: {kernel_error_weight}, repeat_penalty: {repeat_penalty}', 'HEADER')
#     cprint(f'Overlay alpha: {overlay_weight}, Subtle overlay alpha: {subtle_overlay_weight}', 'HEADER')


#     mosaic = Mosaic(
#         source_image=source_image,
#         compare_size=Scale(compare_width, compare_height),
#         tile_size=Scale(tile_width, tile_height),
#         output_size=Scale(output_width, output_height),
#         output_tiles_res=Scale(horizontal_tiles, vertical_tiles),
#         linear_error_weight=linear_error_weight,
#         kernel_error_weight=kernel_error_weight,
#         overlay_alpha=overlay_weight,
#         subtle_overlay_alpha=subtle_overlay_weight,
#         tile_directory=tile_directory,
#         repeat_penalty=repeat_penalty,
#         detail_map=detail_map,
#         subdivisions=subdivisions,
#         subdivision_threshold=args.subdivision_threshold,
#     )
#     mosaic.fit_tiles()
#     mosaic.save(output_path=output_path, show_preview=show_preview)
#     cprint("Done!", 'OKGREEN')





# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PRINT HELPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# -----------------------------------------------------------------------------------------------------------------------------
class Printer:
    """Simple helper for printing progress text."""
    last_line_len = 0
    max_line_len = 0

    load_chars = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
    char_idx = 0

    prntclrs = {
        'GRAY':'\033[90m',
        'DARKBLUE':'\033[34m',
        'DARKMAGENTA':'\033[35m',
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

    def _pad_text(self, text:str) -> str:
        newtext = \
            f"{text}{' ' * (self.last_line_len - len(text))}" \
            if len(text) < self.last_line_len \
            else text
        self.last_line_len = len(text)
        return newtext

    def update_text_len(self):
        self.max_line_len = shutil.get_terminal_size().columns

    def _ensure_length(self, text:str) -> str:
        """Prevent text len from being too long."""
        if len(text) > self.max_line_len:
            return f"{text[:self.max_line_len-3]}..."
        return text

    @staticmethod
    def ctext(text:str, color:str) -> str:
        """Generate a colored string and return it."""
        color = Printer.prntclrs.get(color.upper(), "ENDC")
        return f"{color}{text}{Printer.prntclrs['ENDC']}"

    def cprint(self, text:str, color:str):
        """Print in color (and pad lines to erase old text)"""
        text = self._pad_text(str(text))
        print(self.ctext(text, color))

    def write_progress(self, text):
        """Write status to the terminal without starting a new line, erasing old text."""

        text = f"  {self.next_char()} - {text}..."
        text = self._ensure_length(self._pad_text(text))
        print(self.ctext(text, 'OKCYAN'), end='\r')

# printer is the only way this script prints information.
# so we'll simplify its method calls for readability:
Printer = Printer()
Printer.update_text_len()
cprint = Printer.cprint
ctext = Printer.ctext
cwrite = Printer.write_progress










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



class DebugTimer:
    def __init__(self, text):
        self.text = text
        self.start_time = time.time()
    
    def print(self):
        cprint(
            f"{self.text}: {time.time()-self.start_time:.2f}s",
            "UNDERLINE"
        )



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Scale ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# -----------------------------------------------------------------------------------------------------------------------------
class Scale:
    """A simple helper for bundling width/height information"""
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
    
    def __repr__(self):
        return f"Scale({self.w}x{self.h})"

class InputScale(Scale):
    """Convert an input string into a Scale"""
    def __init__(self, intext:str):
        text = intext.lower().replace(",", "x").replace(".", "x")
        try:
            if "x" in text:
                width, height = text.split("x")
                width, height = int(width), int(height)
            else:
                width = height = int(text)
        except ValueError:
            raise ValueError(f"'{intext}' can't be interpreted as a scale. (Expected format is 'INT' or 'INTxINT')")
        super().__init__(width, height)
            




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ InputTile ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# -----------------------------------------------------------------------------------------------------------------------------
class InputTile:
    compare_size = None
    kernel_error_weight = DEFAULT_KERNEL_WEIGHT
    linear_error_weight = DEFAULT_LINEAR_WEIGHT
    repeat_penalty = DEFAULT_REPEAT_PENALTY

    """A class to hold and work on input tiles"""
    def __init__(self, img, tile_size):
        """Load one image tile, converted to proper output size"""
        if isinstance(img, Image.Image):
            self.img = img
        else:
            self.img = Image.open(img)

        # calculate image cropped size to match tile aspect ratio
        trgt_w, trgt_h = self._crop_from_ratio((self.img.width, self.img.height), tile_size)

        w_delta = self.img.width - trgt_w
        h_delta = self.img.height - trgt_h
        crop = (
            w_delta // 2,
            h_delta // 2,
            self.img.width - (w_delta // 2),
            self.img.height - (h_delta // 2),
        )

         # ONLY resize when the new size is smaller than the og size
        if tile_size.w <= self.img.width and tile_size.h <= self.img.height:
            # crop and resize image to tile size
            self.img = self.img.resize(tile_size, box=crop)
        else:
            self.img = self.img.crop(crop)

        self.img =  self.img.convert(mode='RGB')

        compare_img = self.img.resize(self.compare_size).convert(mode="LAB")
        self.linear_array = self._as_linear_array(compare_img)
        self.kernel_diff_array = self._as_kernel_diff_array(compare_img)
        self.repeat_error = 0.0


    def get_image(self, resize=None):
        """Get the resized tile, and track usage."""
        self.repeat_error += InputTile.repeat_penalty
        return self.img.resize(resize) if resize else self.img


    def compare(self, source) -> float:
        """Get total error score for this tile"""
        err = self.repeat_error
        err += (np.mean(np.abs(source.linear_array - self.linear_array)) / 255) * InputTile.linear_error_weight
        err += (np.mean(np.abs(source.kernel_diff_array - self.kernel_diff_array)) / 255) * InputTile.kernel_error_weight
        return err


    def _as_kernel_diff_array(self, img):
        """
        For each pixel, avg val with neighbors to determine pixel kernel,
        return array representing the differences between the pixels and the pixel kernels.
        """
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


    def _as_linear_array(self, img):
        # convert to Lab color space for more accurate comparisons
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
    subdivisions = DEFAULT_SUBDIVISIONS
    subdivision_threshold = DEFAULT_SUBDIVISION_THRESHOLD
    tile_size = None

    # debug stuff
    min_error = 100.0
    max_error = 0.0
    start_time = None

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
            detail_map,
            subdivisions,
            subdivision_threshold,
    ):
        self.source_image = source_image
        Mosaic.tile_size = tile_size
        InputTile.compare_size = compare_size
        self.output_size = output_size
        self.output_tiles_res = output_tiles_res
        Mosaic.subdivisions = subdivisions
        Mosaic.subdivision_threshold = subdivision_threshold

        if VERBOSE:
            timer = DebugTimer("Setup detail map")

        if detail_map is None:
            self.detail_map = self._make_detail_map()
        else:
            self.detail_map = self._setup_detail_map(detail_map)
        
        if VERBOSE:
            timer.print()
            timer = DebugTimer("Load tiles")

        self.tiles = self.load_tiles(tile_directory)

        if VERBOSE:
            timer.print()

        InputTile.linear_error_weight = linear_error_weight
        InputTile.kernel_error_weight = kernel_error_weight
        InputTile.repeat_penalty = repeat_penalty

        self.overlay_alpha = overlay_alpha
        self.subtle_overlay_alpha = subtle_overlay_alpha

        # create the blank image to create our mosaic
        self.mosaic = Image.new(mode='RGB', size=tuple(self.output_size))


    def _make_detail_map(self):
        """
        Create a default detail map from the source image,
        by running edge detection on the source image, and scaling it down.
        """
        h_tiles, v_tiles = self.output_tiles_res
        map_width = h_tiles * (2 ** self.subdivisions)
        map_height = v_tiles * (2 ** self.subdivisions)
        # generate edge map from the image (Areas with more edges are brighter)
        edge_map = self.source_image\
            .convert(mode="RGB")\
            .filter(ImageFilter.FIND_EDGES)\
            .resize((map_width, map_height))\
            .convert(mode="L")
        # generate a radial gradient with white in the center, black on the edges
        vignette = ImageOps.invert(
            Image.radial_gradient("L").resize((map_width, map_height))
            )
        vignette = ImageEnhance.Contrast(
            ImageChops.overlay(vignette, vignette)
            ).enhance(2)

        # combine edge map and vignette to make a center-biased edge map, as our detail map
        return ImageEnhance.Contrast(
            ImageOps.autocontrast(
                ImageChops.overlay(edge_map, vignette)
                )
            ).enhance(2)


    def _setup_detail_map(self, detail_map):
        """Convert the given detail map into the expected format"""
        h_tiles, v_tiles = self.output_tiles_res
        map_width = h_tiles * (2 ** self.subdivisions)
        map_height = v_tiles * (2 ** self.subdivisions)
        return detail_map\
            .resize((map_width, map_height))\
            .convert(mode="L")


    def load_tiles(self, tile_directory):
        """Open, crop, and rescale tiles from tile directory"""
        tiles = []
        tile_idx = 0
        bad_tile_files = 0
        num_image_tiles = len(os.listdir(tile_directory))
        for img_file in os.scandir(tile_directory):
            tile_idx += 1
            cwrite(f'Loading tile {tile_idx}/{num_image_tiles} ({img_file.name})')
            # PIL will determine what images are or are not valid.
            try:
                tiles.append(InputTile(img_file, self.tile_size))
            except (OSError, ValueError):
                bad_tile_files += 1
                cprint(f"Warning: {img_file.name} could not be loaded", "WARNING")
        cprint(f"{num_image_tiles - bad_tile_files} tiles loaded.", "OKGREEN")
        return tiles


    def find_tile(self, tile_x, tile_y, width, height, sub=0):
        """Find the tile for a single x/y coordinate"""
        # start by finding detail value for this tile
        # decide if we are subdividing based on max detail value in tile
        subdividing = False
        if sub < self.subdivisions:
            detail_map_tile_width = (2**self.subdivisions) // (2**sub)
            crop = (
                detail_map_tile_width * tile_x,
                detail_map_tile_width * tile_y,
                detail_map_tile_width * tile_x + detail_map_tile_width,
                detail_map_tile_width * tile_y + detail_map_tile_width,
            )
            max_detail = np.max(np.array(self.detail_map.crop(crop)))
            if max_detail > self.subdivision_threshold:
                subdividing = True
        
        if subdividing:
            # assemble a composite tile out of sub-tiles
            img = Image.new(mode="RGB", size=(width, height))
            for sub_y in range(2):
                for sub_x in range(2):
                    sub_width, sub_height = width // 2, height // 2
                    sub_crop = (
                        sub_width * sub_x,
                        sub_height * sub_y,
                        sub_width * sub_x + sub_width,
                        sub_height * sub_y + sub_height,
                    )
                    img.paste(
                        self.find_tile(
                            tile_x*2 + sub_x, tile_y*2 + sub_y, 
                            sub_width, sub_height, 
                            sub=sub+1,
                        ),
                        box=sub_crop
                    )
            return img
        # else select one tile the ol'fashioned way

        # crop region for comparison
        crop = (
            tile_x * width,
            tile_y * height,
            tile_x * width + width,
            tile_y * height + height,
        )
        # convert to an InputTile for comparison
        source_region = InputTile(self.source_image.crop(crop), Scale(width, height))

        # scan and find best matching tile image
        final_errors = [tile.compare(source_region) for tile in self.tiles]

        if VERBOSE:
            min_err = min(final_errors)
            max_err = max(final_errors)
            if min_err < self.min_error:
                self.min_error = min_err
            if max_err > self.max_error:
                self.max_error = max_err

        # get the first index where error was equal to the smallest error
        # select the tile at that index
        best_idx = final_errors.index(min(final_errors))
        best_tile = self.tiles[best_idx]
        return self.add_subtle_overlay(best_tile.get_image(resize=(width, height)), source_region.img)


    def fit_tiles(self):
        """
        Scan through horizontal/vertical lines,
        matching a tile to each segment of the source image, 
        and add it to the mosaic.
        """
        horizontal_tiles, vertical_tiles = self.output_tiles_res
        tile_width, tile_height = self.tile_size

        tile_ys = list(range(vertical_tiles))
        tile_xs = list(range(horizontal_tiles))
        random.shuffle(tile_ys)
        total_tiles = horizontal_tiles * vertical_tiles

        tile_idx = 0
        for tile_y in tile_ys:
            random.shuffle(tile_xs)
            for tile_x in tile_xs:

                tile_idx += 1
                cwrite(f"Comparing tile {tile_idx}/{total_tiles} ({tile_x}x{tile_y})")

                # find the tile(s) matching this x/y
                this_tile = self.find_tile(tile_x, tile_y, tile_width, tile_height)
                crop = (
                    tile_x * tile_width,
                    tile_y * tile_height,
                    tile_x * tile_width + tile_width,
                    tile_y * tile_height + tile_height,
                )
                self.mosaic.paste(
                    this_tile, box=crop
                )
                # self.mosaic.paste(
                #     this_tile.get_image(), box=crop
                # )


        cprint(f"{total_tiles} tiles selected.", "OKGREEN")

        if self.overlay_alpha:
            cprint('Applying overlay...', 'OKBLUE')
            self.mosaic = self.add_normal_overlay()
        
        if VERBOSE:
            cprint(f'Smallest error: {self.min_error}, largest error: {self.max_error}', color="UNDERLINE")


    def add_normal_overlay(self):
        overlay_img = ImageChops.overlay(self.mosaic, self.source_image)
        return ImageChops.blend(self.mosaic, overlay_img, self.overlay_alpha)

    def add_subtle_overlay(self, tile, source_tile):
        if not self.subtle_overlay_alpha:
            return tile
        return ImageChops.blend(
            tile, 
            ImageChops.overlay(
                tile,
                source_tile.filter(ImageFilter.GaussianBlur(max(tile.width, tile.height) // 2))
                ), 
            self.subtle_overlay_alpha,
            )


    def save(self, show_preview, output_path):
        """Save (and/or show) the generated mosaic"""
        if show_preview:
            cprint('Showing img...', "OKBLUE")
            self.mosaic.show()

        cprint('Saving img...', "OKBLUE")
        # self.mosaic = self.mosaic.convert(mode="RGB")
        self.mosaic.save(output_path)
        cprint(f'Saved as "{output_path}"', 'OKGREEN')



if __name__ == "__main__":
    main()
