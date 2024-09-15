"""HD Mosaic

A script that creates a photo mosaic by comparing sub-tile pixels for a higher quality result.


Copyright (C) 2024  Ethan Lacasse

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
# ruff: noqa: PLW0603

import argparse
import os
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |                               _    __  __   _   .  .     ___  __                               |
# |                              |  \ |_  |_   /_\  |  | |    |  (__`                              |
# |                              |__/ |__ |   /   \ |__| |__  |  .__)                              |
# |                                                                                                |
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEFAULTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|

DEFAULT_SCALE = 1.0
DEFAULT_COMPARE = '3x3'
DEFAULT_LINEAR_WEIGHT = 1.0
DEFAULT_KERNEL_WEIGHT = 0.1
DEFAULT_OVERLAY = 0.0
DEFAULT_SUBTLE_OVERLAY = 0.5
DEFAULT_REPEAT_PENALTY = 0.1
DEFAULT_SUBDIVISIONS = 1
DEFAULT_SUBDIVISION_THRESHOLD = 150


# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |                                 __       __   __    _        __                                |
# |                                /  _ |   /  \ |__)  /_\  |   (__`                               |
# |                                \__) |__ \__/ |__) /   \ |__ .__)                               |
# |                                                                                                |
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GLOBALS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|

VERBOSE = False
SHOW_PREVIEW = False
USER_INPUT = None
MOSAIC = None


# ||==============================================================================================||
# ||         __     _  _____   _____  _____       ____ __    _ _____   __     _  ________         ||
# ||         ||     | //    ` ||    ` ||   \\      ||  |\\   | ||   \\ ||     | '   ||   '        ||
# ||         ||     | \\___   ||___   ||___//      ||  | \\  | ||___// ||     |     ||            ||
# ||         ||     |      \\ ||      || \\        ||  |  \\ | ||      ||     |     ||            ||
# ||          \\___/  \____// ||____, ||  \\,     _||_ |   \\| ||       \\___/     _||_           ||
# ||                                                                                              ||
# ||==============================================================================================||
# ||========================================= USER INPUT =========================================||
# ||==============================================================================================||


#                      ___                  __
#                       |   _  ,_      _|_ |__)  __╮ ,_  __╮ ,  ,  _  _|_  _  ,_
#                      _|_ | | |_) (_|  |  |    (_/| |  (_/| |\/| (/,  |  (/, |
#                              j
#                      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class InputParameter:
    """Stores information about an input parameter"""

    def __init__(
            self,
            name: str,
            *,
            help: str = '',
            type: callable = str,
            prompt: str|None = None,
            metavar: str|None = None,
            default=None,
            static: bool = False,
            allow_none: bool = False):
        """Create an InputParameter object for handling `input` options."""
        self.name = name
        self.help = help
        self.prompt = f'{name}: {help}' if prompt is None else prompt
        self.type = type
        self.metavar = metavar
        self.default = default if default is None else type(default)
        self.value = default if default is None else type(default)
        self.static = static
        self.allow_none = allow_none

    def set_val(self, val):
        """Assign a value to this parameter"""
        if val is None and self.allow_none:
            self.value = val
        else:
            self.value = self.type(val)

    def prompt_string(self, add_in: str|None = None) -> str:
        """Get a printable string prompting the user to provide a new value."""
        string = f"\n\n{ctext(ctext(self.name, 'HEADER'), 'BOLD')} : {ctext(self.metavar, 'GRAY')}"
        if self.value is not None:
            string += f" = {ctext(str(self.value), 'GRAY')}"
        if self.default is not None:
            string += ctext(f' (default: {self.default})', 'GRAY')
        string += ctext(f'\n{self.help}', 'gray')
        string += ctext(f'\n\n{self.prompt}', 'OKBLUE')
        if add_in:
            string += f'\n\n{add_in}'
        else:
            string += '\n\n'
        string += ctext(f'\n{self.name} = ', 'HEADER')
        return string


#                             ___                   _
#                              |   _  ,_      _|_  /_\   _  _|_ °  _   _
#                             _|_ | | |_) (_|  |  /   \ (_,  |  | (_) | |
#                                     j
#                             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class InputAction:
    """Stores a callback for UserInterface"""

    def __init__(
            self,
            name: str,
            description: str,
            callback: callable):
        """Create a new InputAction with given name, description, and callback."""
        self.name = name
        self.description = description
        self.callback = callback


#                          .  .           ___                 _
#                          |  |  _  _  ,_  |   _  _|_  _  ,_ /_  __╮  _   _
#                          |__| _\ (/, |  _|_ | |  |  (/, |  |  (_/| (_, (/,
#                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class UserInterface:
    """Gets input variables from the user, creating a simple menu."""

    def __init__(self, name: str):
        """Create an object for managing a simple CLI menu."""
        self.name = name
        self.categories = {}
        self.params = {}
        self.actions = {}
        self.option_callback = None

    def __getitem__(self, key):
        return self.params[key]

    @staticmethod
    def _clear_screen():
        size = shutil.get_terminal_size()
        print('\n' * size.lines)

    def add_category(self, category: str, description: str):
        """Add a new submenu named "category", which organizes some parameters."""
        self.categories[category] = {'desc': description, 'params': []}

    def add_parameter(
            self,
            name,
            category='',
            **kwargs):
        """Add a new parameter to the menu."""
        param = InputParameter(name, **kwargs)
        self.categories[category]['params'].append(name)
        self.params[name] = param

    def add_action(
            self,
            name,
            desc,
            callback):
        """Add a new action to the menu."""
        self.actions[name] = InputAction(name, desc, callback)

    def get_param(self, name: str, prev_message: str = '') -> str:
        """Prompt the user to get a new value for parameter."""
        param = self[name]

        add_in = None

        # get user input for this param, and set it
        # if there is an error, print it nicely and keep trying:
        while True:
            try:
                # clear screen
                self._clear_screen()

                print(prev_message)

                inpt = input(param.prompt_string(add_in=add_in))
                inpt = self.pre_process_input(inpt)
                param.set_val(inpt)
                if self.option_callback is not None:
                    self.option_callback(name)
                return ctext(f'Set {param.name} to {param.value}.', 'OKGREEN')
            except (ValueError) as e:
                add_in = ctext(getattr(e, 'message', str(e)), 'WARNING')

    def get_static(self):
        """Prompt the user to fill all static parameters"""
        prev_message = ''
        for name, param in self.params.items():
            if param.static:
                prev_message = self.get_param(name, prev_message=prev_message)
        if prev_message:
            print(prev_message)

    @staticmethod
    def pre_process_input(inpt: str) -> str:
        """Clean input, and also handle exit command"""
        inpt = inpt.strip()
        if inpt == 'exit':
            sys.exit()
        return inpt

    def _print_main_menu(self, ex_txt: str|None = None):
        # print the menu options
        self._clear_screen()

        cprint(self.name, 'OKBLUE')
        cprint('\nActions:', 'GRAY')

        for idx, items in enumerate(self.actions.items()):
            name, action = items
            print(f'{ctext(str(idx), "HEADER")}:  {ctext(name, "OKCYAN")} {ctext(f"- {action.description}", "GRAY")}')

        cprint('\nOptions:', 'GRAY')

        for idx, item in enumerate(self.categories.items(), start=len(self.actions)):
            category, vals = item
            desc = vals['desc']
            print(f'{ctext(str(idx), "HEADER")}: {ctext(category, "HEADER")} {ctext(f"- {desc}", "GRAY")}')

        if ex_txt:
            print('\n' + ex_txt)
        else:
            print('\n')
        cprint('Select an action or option category:', 'GRAY')

    def _print_category_menu(self, category: str, ex_txt: str|None = None):
        # print the menu options
        self._clear_screen()

        desc = self.categories[category]['desc']
        params = self.categories[category]['params']

        cprint(category, 'OKBLUE')
        cprint(desc, 'GRAY')

        cprint('\n\nOptions:', 'GRAY')

        for idx, param in enumerate(params):
            print(
                f'{ctext(str(idx), "HEADER")}{ctext(":", "GRAY")}'
                f'{ctext(param, "HEADER")}{ctext(" = ", "GRAY")}'
                f'{ctext(repr(self[param].value), "DARKBLUE")}'
                f'{ctext(f" - {self[param].help}", "GRAY")}',
                )

        if ex_txt:
            print('\n' + ex_txt)
        else:
            print('\n')
        cprint('Select an option:', 'GRAY')

    def category_menu(self, category, ex_txt=None):
        """Show the menu for this category"""
        # ex txt holds feedback info to display above the menu
        params = self.categories[category]['params']
        while True:
            # keep running the menu until back is given
            self._print_category_menu(category, ex_txt=ex_txt)
            choice = self.pre_process_input(input())
            choice = self.filter_choice(choice, params)
            if choice in {'back', '..', '-'}:
                return
            if choice in params:
                ex_txt = self.get_param(choice)
            elif choice.isnumeric() and int(choice) < len(params):
                param = params[int(choice)]
                ex_txt = self.get_param(param)
            else:
                ex_txt = ctext(f"'{choice}' isn't a valid choice.", 'WARNING')

    def main_menu(self, ex_txt: str|None = None):
        """Show the main menu"""
        actions = list(self.actions.keys())
        categories = list(self.categories.keys())
        all_options = actions + categories
        while True:
            self._print_main_menu(ex_txt=ex_txt)
            choice = self.pre_process_input(input())
            choice = self.filter_choice(choice, all_options)
            if choice in self.actions:
                ex_txt = self.actions[choice].callback()
            elif choice in self.categories:
                self.category_menu(choice)
                ex_txt = None
            else:
                ex_txt = ctext(f"'{choice}' isn't a valid choice.", 'WARNING')

    @staticmethod
    def filter_choice(choice: str, options: list) -> str:
        """Try fitting choice to option, so user input doesn't need to be so specific."""
        if choice in options:
            return choice
        if choice.isnumeric() and int(choice) < len(options):
            return options[int(choice)]
        # finally, try fuzzy matching choice to options
        options = [opt for opt in options if opt.lower().startswith(choice.lower())]
        print(options)
        if len(options) == 1:
            return options[0]
        return choice

    def verify_all_filled(self):
        """Force all values to be filled (except for those that are allowed to be None)"""
        for name, param in self.params.items():
            if param.value is None and not param.allow_none:
                self.get_param(name)


# ||==============================================================================================||
# ||     __      __          ____ __    _      _____    ____  _____   ____ _____    ________      ||
# ||     |\\    /||    /\     ||  |\\   |     //    `  //   ` ||   \\  ||  ||   \\ '   ||   '     ||
# ||     | \\  / ||   / \\    ||  | \\  |     \\___   ||      ||___//  ||  ||___//     ||         ||
# ||     |  \\/  ||  /___\\   ||  |  \\ |          \\ ||      || \\    ||  ||          ||         ||
# ||     |       || /     \\ _||_ |   \\|     \____//  \\___/ ||  \\, _||_ ||         _||_        ||
# ||                                                                                              ||
# ||==============================================================================================||
# ||========================================= MAIN SCRIPT ========================================||
# ||==============================================================================================||


# :------------------------------------------------------------------------------------------------:
# :                 ___ ,  , ___ ___ ___   _       ___ ___    _   ___ ___  __  ,  ,                :
# :                  |  |\ |  |   |   |   /_\  |    |   _/   /_\   |   |  /  \ |\ |                :
# :                 _|_ | \| _|_  |  _|_ /   \ |__ _|_ /__, /   \  |  _|_ \__/ | \|                :
# :                                                                                                :
# :------------------------------------------------------------------------------------------------:
# :---------------------------------------- Initialization ----------------------------------------:
# :------------------------------------------------------------------------------------------------:

def init_parser():
    """Get argparse arguments."""
    global VERBOSE
    # argparser for verbose and help messages
    parser = argparse.ArgumentParser(
        description=ctext(
            'This tool can be used to create a high-quality image mosaic '
            'by comparing given image tiles to a source image.',
            'DARKBLUE',
            ),
    )
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    VERBOSE = args.verbose


def init_ui() -> UserInterface:
    """Create and initialize the UserInterface object

    Returns:
        UserInterface
    """

    ui = UserInterface('hd_mosaic')

    ui.add_category('input/output', 'Options for the input/output images')
    ui.add_category('tiles', 'Options relating to the tiles')
    ui.add_category('weights', 'Weights to use for different comparison methods')
    ui.add_category('overlay', 'Options relating to the image overlay')
    ui.add_category('subdivision', 'Options relating to the tile subdivision')
    ui.add_category('other', 'Uncategorized options')

    ui.add_parameter(
        'tile_folder', category='tiles',
        help='The source folder containing all the image tiles.',
        prompt='Please provide the path to a folder of images to use as tiles:',
        metavar='PATH', type=folder_path,
        )
    ui.add_parameter(
        'tile_load_resolution', category='tiles',
        help='The resolution to load the tiles in with.',
        prompt="""Please provide the resolution you would like to load the tiles in with:
(Small resolutions might look bad, big resolutions might be very slow. 128x128 is an okay starting point.)""",
        metavar='INT|INTxINT', type=InputScale,
        )
    ui.add_parameter(
        'xy_tiles', category='tiles',
        help='The number of horizontal and vertical tiles to use in the output mosaic.',
        prompt='Please provide the number of horizontal/vertical tiles you would like to use:',
        metavar='INT|INTxINT', type=InputScale,
        )
    ui.add_parameter(
        'source_image', category='input/output',
        help='The source image to base the mosaic on.',
        prompt='Provide the path to an image to base this mosaic on:',
        metavar='PATH', type=file_path,
        )
    ui.add_parameter(
        'output', category='input/output',
        help='The source image to base the mosaic on.',
        prompt='Provide the path to an image to base this mosaic on:',
        metavar='PATH', allow_none=True,
        )
    ui.add_parameter(
        'input_rescale', category='input/output',
        help='Multiplier to rescale source image by.',
        prompt='Enter a rescale amount for the source_image:',
        metavar='FLOAT', default=DEFAULT_SCALE, type=float,
    )
    ui.add_parameter(
        'compare_res', category='tiles',
        help='The resolution that tiles will be compared at.',
        prompt='Enter a new compare scale:',
        metavar='INT|INTxINT', default=DEFAULT_COMPARE, type=InputScale,
    )
    ui.add_parameter(
        'linear_error_weight', category='weights',
        help="How much the 'linear' comparison affects the output.",
        prompt='Enter a new linear weight:',
        metavar='FLOAT', default=DEFAULT_LINEAR_WEIGHT, type=float,
    )
    ui.add_parameter(
        'kernel_error_weight', category='weights',
        help="How much the 'kernel difference' comparion affects the output.",
        prompt='Enter a new linear weight:',
        metavar='FLOAT', default=DEFAULT_KERNEL_WEIGHT, type=float,
    )
    ui.add_parameter(
        'overlay_opacity', category='overlay',
        help="The alpha for a 'normal' overlay of the target image over the mosaic.",
        prompt='Enter a new overlay alpha:',
        metavar='FLOAT', default=DEFAULT_OVERLAY, type=float,
    )
    ui.add_parameter(
        'subtle_overlay', category='overlay',
        help='The alpha for an alternate, less sharp method of overlaying the target image on the mosaic.',
        prompt='Enter a new subtle overlay alpha:',
        metavar='FLOAT', default=DEFAULT_SUBTLE_OVERLAY, type=float,
    )
    ui.add_parameter(
        'repeat_penalty', category='weights',
        help='How much to penalize repetition when selecting tiles.',
        prompt='Enter a new repetition penalty:',
        metavar='FLOAT', default=DEFAULT_REPEAT_PENALTY, type=float,
    )
    ui.add_parameter(
        'show', category='other',
        help='If True, opens a preview of the output image upon completion.',
        prompt="Enter True or False to enable/disable 'show':",
        metavar='BOOL', default=False, type=bool,
    )
    ui.add_parameter(
        'subdivisions', category='subdivision',
        help='Max number of subdivisions allowed in each main tile.',
        prompt='Enter the new number of subdivisions:',
        metavar='INT', default=DEFAULT_SUBDIVISIONS, type=int,
    )
    ui.add_parameter(
        'subdivision_threshold', category='subdivision',
        help='Detail values higher than this threshold will create a subdivision.',
        prompt='Enter a new subdiv theshold:',
        metavar='INT', default=DEFAULT_SUBDIVISION_THRESHOLD, type=int,
    )
    ui.add_parameter(
        'detail_map', category='subdivision',
        help='An image that controls where extra subdivisions are added.',
        prompt='Enter the new number of subdivisions:',
        metavar='PATH', allow_none=True, type=file_path,
    )
    return ui


def set_cwd():
    """Fix issue where CWD might be in System32

    If you run the script by double clicking the file in windows,
    the CWD might be in System32 instead of in the folder where the script is located.
    To fix this, we can just check if this is the CWD on startup and set it to the script location.
    """
    if 'system32' in os.getcwd().lower():
        try:
            os.chdir(os.path.split(__file__)[0])
        except (NameError, OSError) as e:
            if VERBOSE:
                print(e)


# :------------------------------------------------------------------------------------------------:
# :                            __    _            __    _    __       __                           :
# :                           /  `  /_\  |   |   |__)  /_\  /  ` |_/ (__`                          :
# :                           \__, /   \ |__ |__ |__) /   \ \__, | \ .__)                          :
# :                                                                                                :
# :------------------------------------------------------------------------------------------------:
# :------------------------------------------- Callbacks ------------------------------------------:
# :------------------------------------------------------------------------------------------------:

def load_options_into_mosaic(modified_option):
    """Load given option from USER_INPUT into MOSAIC"""
    global SHOW_PREVIEW
    val = USER_INPUT[modified_option].value

    match modified_option:
        case 'source_image':
            MOSAIC.config(source_image_path=val)
        case 'xy_tiles':
            MOSAIC.config(output_tiles_res=val)
        case 'input_rescale':
            MOSAIC.config(source_image_rescale=val)
        case 'overlay_opacity':
            MOSAIC.config(overlay_alpha=val)
        case 'subtle_overlay':
            MOSAIC.config(subtle_overlay_alpha=val)
        case 'detail_map':
            MOSAIC.config(detail_map_path=val)
        case 'show':
            SHOW_PREVIEW = val
        case 'tile_folder':
            MOSAIC.config(tile_load_dir=val)
        case 'output':
            # avoid any keys not used by MOSAIC
            return
        case _:
            MOSAIC.config(**{modified_option: val})


def make_mosaic() -> str:
    """Make a mosaic using the Mosaic class."""
    USER_INPUT.verify_all_filled()
    MOSAIC.fit_tiles()
    output_path = USER_INPUT['output'].value
    if output_path is None:
        output_path = gen_output_path()
    MOSAIC.save(output_path=output_path, show_preview=SHOW_PREVIEW)
    return ctext(f'Saved as "{output_path}"', 'OKGREEN')


# :------------------------------------------------------------------------------------------------:
# :                                 _  _  __      __   __  __   __                                 :
# :                                 |__| |_  |   |__) |_  |__) (__`                                :
# :                                 |  | |__ |__ |    |__ | \_ .__)                                :
# :                                                                                                :
# :------------------------------------------------------------------------------------------------:
# :-------------------------------------------- Helpers -------------------------------------------:
# :------------------------------------------------------------------------------------------------:

def gen_output_path() -> str:
    """Generate a default output filename."""
    output_name = 'mosaic'
    for param_name, symbol in [
        ('xy_tiles', ''),
        ('input_rescale', 'S'),
        ('compare_res', 'c'),
        ('linear_error_weight', 'l'),
        ('kernel_error_weight', 'k'),
        ('overlay_opacity', 'o'),
        ('subtle_overlay', 's'),
        ('repeat_penalty', 'r'),
        ('subdivisions', 'd'),
        ('subdivision_threshold', 't'),
    ]:
        param = USER_INPUT[param_name]
        if param.value != param.default:
            output_name += f'_{symbol}{param.value}'
    output_name += '.jpg'
    return output_name


def file_path(path: str) -> str:
    """Verify (and clean) a path to an image

    Raises:
        ValueError: When path is not valid
    """
    # remove filepath quotes
    for quote in ("'", '"'):
        if path.startswith(quote) and path.endswith(quote):
            path = path.removeprefix(quote).removesuffix(quote)

    if os.path.exists(path) and os.path.isfile(path):
        return path
    msg = f"Couldn't find a file at '{path}'. Make sure this is a valid path to an image."
    raise ValueError(msg)


def folder_path(path: str) -> str:
    """Verify (and clean) a path to a folder

    Raises:
        ValueError: when path isn't correct.
    """
    # remove filepath quotes
    for quote in ("'", '"'):
        if path.startswith(quote) and path.endswith(quote):
            path = path.removeprefix(quote).removesuffix(quote)

    if os.path.exists(path) and os.path.isdir(path):
        return path
    msg = f"Couldn't find a folder at '{path}'. Make sure this is a valid path to a folder of images."
    raise ValueError(msg)


def crop_from_rescale(old_size, new_size) -> tuple[int, int, int, int]:
    """Create a crop for a scale that maintains the aspect ratio"""
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


# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |                                   __      __                                                   |
# |                                   |\\    /||  __    °                                          |
# |                                   | \\  / ||  __\   ╮    __                                    |
# |                                   |  \\/  || /   |  |  ╮/  |                                   |
# |                                   |       || \__/|, |, |   |,                                  |
# |                                                                                                |
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ main ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|

def main():
    """Run the main script."""
    global USER_INPUT, MOSAIC

    # setup parser and ui
    set_cwd()
    init_parser()
    ui = USER_INPUT = init_ui()

    # add mosaic action to ui
    ui.add_action('Make_Mosaic', 'Generate the output mosaic', make_mosaic)

    # get static params, use those to configure a new Mosaic
    MOSAIC = Mosaic()
    ui.option_callback = load_options_into_mosaic
    # load all default vals
    for name, param in ui.params.items():
        if param.value is not None:
            load_options_into_mosaic(name)
    # start the main menu of the program
    ui.main_menu()


# ||==============================================================================================||
# || __    __ _____ __    _____   _____ _____     ____ __             _____  _____  _____  _____  ||
# || ||    ||||    `||    ||   \\||    `||   \\  //   `||       /\   //    `//    `||    `//    ` ||
# || ||____||||___  ||    ||___//||___  ||___// ||     ||      / \\  \\___  \\___  ||___  \\___   ||
# || ||    ||||     ||    ||     ||     || \\   ||     ||     /___\\      \\     \\||          \\ ||
# || ||    ||||____,||___,||     ||____,||  \\,  \\___/||___,/     \\\____//\____//||____,\____// ||
# ||                                                                                              ||
# ||==============================================================================================||
# ||======================================  Helper Classes  ======================================||
# ||==============================================================================================||


# :------------------------------------------------------------------------------------------------:
# :                                  __   __  ___ ,  , ___  __  __                                 :
# :                                 |__) |__)  |  |\ |  |  |_  |__)                                :
# :                                 |    | \_ _|_ | \|  |  |__ | \_                                :
# :                                                                                                :
# :------------------------------------------------------------------------------------------------:
# :-------------------------------------------- Printer -------------------------------------------:
# :------------------------------------------------------------------------------------------------:
class Printer:
    """Simple helper for printing progress text."""

    _last_line_len = 0
    _max_line_len = 0

    _load_chars = ['⢿', '⣻', '⣽', '⣾', '⣷', '⣯', '⣟', '⡿']
    _char_idx = 0

    _prntclrs = {
        'GRAY': '\033[90m',
        'DARKBLUE': '\033[34m',
        'DARKMAGENTA': '\033[35m',
        'HEADER': '\033[95m',
        'OKBLUE': '\033[94m',
        'OKCYAN': '\033[96m',
        'OKGREEN': '\033[92m',
        'WARNING': '\033[93m',
        'FAIL': '\033[91m',
        'ENDC': '\033[0m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
    }

    def next_char(self) -> str:
        """Get the next loading character"""
        self._char_idx = (self._char_idx + 1) % len(self._load_chars)
        return self._load_chars[self._char_idx]

    def _pad_text(self, text: str) -> str:
        newtext = \
            f"{text}{' ' * (self._last_line_len - len(text))}" \
            if len(text) < self._last_line_len \
            else text
        self._last_line_len = len(text)
        return newtext

    def update_text_len(self):
        """Update printer max text len (based on terminal size)"""
        self._max_line_len = shutil.get_terminal_size().columns

    def _ensure_length(self, text: str) -> str:
        """Prevent text len from being too long."""
        if len(text) > self._max_line_len:
            return f'{text[:self._max_line_len - 3]}...'
        return text

    @classmethod
    def ctext(cls, text: str, color: str) -> str:
        """Generate a colored string and return it."""
        color = cls._prntclrs.get(color.upper(), 'ENDC')
        return f"{color}{text}{cls._prntclrs['ENDC']}"

    def cprint(self, text: str, color: str):
        """Print in color (and pad lines to erase old text)"""
        text = self._pad_text(str(text))
        print(self.ctext(text, color))

    def write_progress(self, text):
        """Write status to the terminal without starting a new line, erasing old text."""
        text = f'  {self.next_char()} - {text}...'
        text = self._ensure_length(self._pad_text(text))
        print(self.ctext(text, 'OKCYAN'), end='\r')


# printer is the main way this script prints information.
# so we'll simplify its method calls for readability:
Printer = Printer()
Printer.update_text_len()
cprint = Printer.cprint
ctext = Printer.ctext
cwrite = Printer.write_progress


# :------------------------------------------------------------------------------------------------:
# :                          _    __  __  .  .  __  ___ ___ ,   ,  __  __                          :
# :                         |  \ |_  |__) |  | /  _  |   |  |\_/| |_  |__)                         :
# :                         |__/ |__ |__) |__| \__)  |  _|_ |   | |__ | \_                         :
# :                                                                                                :
# :------------------------------------------------------------------------------------------------:
# :------------------------------------------ DebugTimer ------------------------------------------:
# :------------------------------------------------------------------------------------------------:
class DebugTimer:
    """Simple helper for timing various operations"""

    def __init__(self, text: str):
        """Create a DebugTimer that times a task named `text`"""
        self.text = text
        self.start_time = time.time()

    def print(self):
        """Print the result of the timer"""
        cprint(
            f'{self.text}: {time.time() - self.start_time:.2f}s',
            'UNDERLINE',
        )


# :------------------------------------------------------------------------------------------------:
# :                                      __   __    _        __                                    :
# :                                     (__` /  `  /_\  |   |_                                     :
# :                                     .__) \__, /   \ |__ |__                                    :
# :                                                                                                :
# :------------------------------------------------------------------------------------------------:
# :--------------------------------------------- Scale --------------------------------------------:
# :------------------------------------------------------------------------------------------------:
class Scale:
    """A simple helper for bundling width/height information"""

    def __init__(self, width: int, height: int):
        """Create a Scale with given width and height"""
        self.w = width
        self.h = height

    def __iter__(self):
        yield from (self.w, self.h)

    def __len__(self):
        return 2

    def __eq__(self, other):
        if isinstance(other, str):
            return f'{self.w}x{self.h}' == other.lower()
        if isinstance(other, tuple) or hasattr(other, '__iter__'):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self))

    def __getitem__(self, idx):
        if idx == 0:
            return self.w
        return self.h

    def __repr__(self):
        return f'{self.w}x{self.h}'


class InputScale(Scale):
    """Convert an input string into a Scale"""

    def __init__(self, intext: str):
        """Create a Scale from an input str

        Raises:
            ValueError: When string can't be understood as a height/width
        """
        text = intext.lower().replace(',', 'x').replace('.', 'x')
        try:
            if 'x' in text:
                width, height = text.split('x')
                width, height = int(width), int(height)
            else:
                width = height = int(text)
        except ValueError:
            msg = f"'{intext}' can't be interpreted as a scale. (Expected format is 'INT' or 'INTxINT')"
            raise ValueError(msg)  # noqa: B904
        if width == 0 or height == 0:
            msg = f'{width}x{height} is too small. Values smaller than 1 are impossible.'
            raise ValueError(msg)
        super().__init__(width, height)


# ||==============================================================================================||
# ||            ____ __    _ _____   __     _  ________   ________  ____ __      _____            ||
# ||             ||  |\\   | ||   \\ ||     | '   ||   ' '   ||   '  ||  ||     ||    `           ||
# ||             ||  | \\  | ||___// ||     |     ||         ||      ||  ||     ||___             ||
# ||             ||  |  \\ | ||      ||     |     ||         ||      ||  ||     ||                ||
# ||            _||_ |   \\| ||       \\___/     _||_       _||_    _||_ ||___, ||____,           ||
# ||                                                                                              ||
# ||==============================================================================================||
# ||========================================== InputTile =========================================||
# ||==============================================================================================||
class InputTile:
    """Load/store images and make comparisons between them"""

    compare_res = None
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

        self.img = self.img.convert(mode='RGB')

        self.linear_array = None
        self.kernel_diff_array = None
        self.repeat_error = 0.0

    def setup_compare_arrays(self):
        """Create numpy arrays for comparing image similarity"""
        compare_img = self.img.resize(self.compare_res).convert(mode='LAB')
        self.linear_array = self._as_linear_array(compare_img)
        self.kernel_diff_array = self._as_kernel_diff_array(compare_img)

    def get_image(self, resize=None) -> Image.Image:
        """Get the resized tile, and track usage."""
        self.repeat_error += InputTile.repeat_penalty
        return self.img.resize(resize) if resize else self.img

    def compare(self, source) -> float:
        """Get total error score for this tile"""
        err = self.repeat_error
        err += (np.mean(np.abs(source.linear_array - self.linear_array)) / 255) * InputTile.linear_error_weight
        err += (
            np.mean(np.abs(source.kernel_diff_array - self.kernel_diff_array)) / 255
            ) * InputTile.kernel_error_weight
        return err

    @staticmethod
    def _as_kernel_diff_array(img) -> np.array:
        """For each pixel, avg val with neighbors to determine pixel kernel.

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
                arr = np.append(arr, abs(img.getpixel((x, y))[0] - avg_val))
        return arr

    @staticmethod
    def _as_linear_array(img) -> np.array:
        """Convert Image to an array for comparison"""
        # convert to Lab color space for more accurate comparisons
        return np.array(
            [img.getpixel((x, y)) for y in range(img.height) for x in range(img.width)],
            )

    @staticmethod
    def _crop_from_ratio(width_height, ratio) -> Scale:
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


# ||==============================================================================================||
# ||                       __      __   ____    _____           ____   ____                       ||
# ||                       |\\    /||  //  \\  //    `    /\     ||   //   `                      ||
# ||                       | \\  / || ||    || \\___     / \\    ||  ||                           ||
# ||                       |  \\/  || ||    ||      \\  /___\\   ||  ||                           ||
# ||                       |       ||  \\__//  \____// /     \\ _||_  \\___/                      ||
# ||                                                                                              ||
# ||==============================================================================================||
# ||=========================================== Mosaic ===========================================||
# ||==============================================================================================||
class Mosaic:
    """A class to hold and work on the mosaic tiles."""

    tile_load_resolution = None
    arrays_made = False

    # debug stuff
    min_error = 100.0
    max_error = 0.0

    def __init__(self):
        """Create a Mosaic, loading images from given directory using given size."""
        self.tile_load_resolution = None
        self.tile_load_dir = None
        self.tile_size = None
        self.source_image = None
        self.source_image_path = None
        self.source_image_rescale = None
        self.compare_res = None
        self.output_size = None
        self.output_tiles_res = None
        self.linear_error_weight = None
        self.kernel_error_weight = None
        self.overlay_alpha = None
        self.subtle_overlay_alpha = None
        self.repeat_penalty = None
        self.detail_map = None
        self.detail_map_path = None
        self.auto_detail_map = True
        self.subdivisions = None
        self.subdivision_threshold = None
        self.tiles = None

    def load_tiles(self):
        """Open, crop, and rescale tiles from tile directory"""
        if VERBOSE:
            timer = DebugTimer('Load Tiles')

        # remember that these tiles haven't had arrays made yet
        Mosaic.arrays_made = False

        self.tiles = tiles = []
        bad_tile_files = 0
        num_image_tiles = len(os.listdir(self.tile_load_dir))

        for tile_idx, img_file in enumerate(os.scandir(self.tile_load_dir)):
            cwrite(f'Loading tile {tile_idx}/{num_image_tiles} ({img_file.name})')
            # PIL will determine what images are or are not valid.
            try:
                tiles.append(InputTile(img_file, self.tile_load_resolution))
            except (OSError, ValueError):
                bad_tile_files += 1
                cprint(f'Warning: {img_file.name} could not be loaded', 'WARNING')

        cprint(f'{num_image_tiles - bad_tile_files} tiles loaded.', 'OKGREEN')

        if VERBOSE:
            timer.print()

    def make_all_compare_arrays(self):
        """Create 'compare' arrays for each tile in Mosaic"""
        Mosaic.arrays_made = True
        num_tiles = len(self.tiles)
        for idx, tile in enumerate(self.tiles):
            tile.setup_compare_arrays()
            cwrite(f'Making compare array {idx}/{num_tiles}')

    def config(self, **kwargs):
        """Update configuraiton for Mosaic

        kwargs:
            source_image_path
            source_image_rescale
            compare_res
            output_tiles_res
            subdivisions
            subdivision_threshold
            detail_map_path
            linear_error_weight
            kernel_error_weight
            repeat_penalty
            overlay_alpha
            subtle_overlay_alpha
            tile_load_dir
            tile_load_resolution

        Raises:
            ValueError: When a given keyword is not an extant attribute of Mosaic
        """
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
            else:
                msg = f'"{key}" is not an attribute of {self}'
                raise ValueError(msg)
        if 'compare_res' in kwargs:
            InputTile.compare_res = kwargs['compare_res']
            # only regen arrays immediately if they have already been generated
            if self.arrays_made:
                self.make_all_compare_arrays()

        # can we load in the input tiles with the settings we have?
        tile_input_changed = ('tile_load_resolution' in kwargs) or ('tile_load_dir' in kwargs)
        can_load_tiles = (self.tile_load_resolution is not None) and (self.tile_load_dir is not None)
        if tile_input_changed and can_load_tiles:
            self.load_tiles()

        # have our settings changed in a way that require us to redo some setup?
        input_output_changed = (
            tile_input_changed
            or 'source_image_path' in kwargs
            or 'source_image_rescale' in kwargs
            or 'output_tiles_res' in kwargs
            or 'subdivisions' in kwargs
            or 'detail_map_path' in kwargs
            )
        # do we have enough information to do setup?
        can_setup_source = (
            can_load_tiles
            and self.source_image_path is not None
            and self.output_tiles_res is not None
        )
        if input_output_changed and can_setup_source:
            self._setup_source()
            # create the blank image to create our mosaic
            self.mosaic = Image.new(mode='RGB', size=tuple(self.output_size))
            # create a default, or setup the provided, detail map
            if self.detail_map_path is None:
                self.detail_map = self._make_detail_map()
            else:
                self.detail_map = self._setup_detail_map()

    def _setup_source(self):
        """Load input image and setup adjusted image scales"""
        # parse image scale
        rescale_by = self.source_image_rescale
        source_path = self.source_image_path
        source_image = Image.open(source_path)

        cprint(f'Target image input size: {source_image.width}x{source_image.height}', 'HEADER')

        if rescale_by != 1.0:
            source_image = source_image.resize((
                int(source_image.width * rescale_by),
                int(source_image.height * rescale_by),
            ))
            cprint(f'Rescaled to: {source_image.width}x{source_image.height}', 'HEADER')

        # tile size must divide evenly into subdivision width
        subdivisions = self.subdivisions
        horizontal_tiles, vertical_tiles = self.output_tiles_res
        sub_width = (2**subdivisions)
        tile_width = (source_image.width // (horizontal_tiles * sub_width)) * sub_width
        tile_height = (source_image.height // (vertical_tiles * sub_width)) * sub_width
        self.tile_size = Scale(tile_width, tile_height)

        # calculate real output size (for equally sized tiles)
        output_width = tile_width * horizontal_tiles
        output_height = tile_height * vertical_tiles
        self.output_size = Scale(output_width, output_height)

        # resize input image for exact comparison with tiles
        og_width, og_height = source_image.width, source_image.height
        crop = crop_from_rescale((og_width, og_height), (output_width, output_height))
        source_image = source_image.resize((output_width, output_height), box=crop)
        self.source_image = source_image.convert(mode='RGB')

        cprint(
            f'Output tiles: {horizontal_tiles}x{vertical_tiles} tiles, {tile_width}x{tile_height}px each.',
            'HEADER',
            )
        cprint(
            f'With {subdivisions} subdivision steps, '
            f'up to {horizontal_tiles * sub_width}x{vertical_tiles * sub_width} total sub-tiles.',
            'HEADER',
            )
        cprint(f'Output image size: {self.source_image.width}x{self.source_image.height}', 'HEADER')

    def _make_detail_map(self) -> Image.Image:
        """Create a default detail map from the source image.

        Creates a detail map by running edge detection on the source image, and scaling it down.
        """
        h_tiles, v_tiles = self.output_tiles_res
        map_width = h_tiles * (2 ** self.subdivisions)
        map_height = v_tiles * (2 ** self.subdivisions)
        # generate edge map from the image (Areas with more edges are brighter)
        edge_map = self.source_image\
            .convert(mode='RGB')\
            .filter(ImageFilter.FIND_EDGES)\
            .resize((map_width, map_height))\
            .convert(mode='L')
        # generate a radial gradient with white in the center, black on the edges
        vignette = ImageOps.invert(
            Image.radial_gradient('L').resize((map_width, map_height)),
            )
        vignette = ImageEnhance.Contrast(
            ImageChops.overlay(vignette, vignette),
            ).enhance(2)

        # combine edge map and vignette to make a center-biased edge map, as our detail map
        return ImageEnhance.Contrast(
            ImageOps.autocontrast(
                ImageChops.overlay(edge_map, vignette),
                ),
            ).enhance(2)

    def _setup_detail_map(self) -> Image.Image:
        """Convert the given detail map into the expected format"""
        detail_map = Image.open(self.detail_map_path)
        h_tiles, v_tiles = self.output_tiles_res
        map_width = h_tiles * (2 ** self.subdivisions)
        map_height = v_tiles * (2 ** self.subdivisions)
        return detail_map\
            .resize((map_width, map_height))\
            .convert(mode='L')

    def find_tile(self, tile_x, tile_y, width, height, sub=0) -> Image.Image:
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
            img = Image.new(mode='RGB', size=(width, height))
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
                            tile_x * 2 + sub_x, tile_y * 2 + sub_y,
                            sub_width, sub_height,
                            sub=sub + 1,
                        ),
                        box=sub_crop,
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
        source_region.setup_compare_arrays()

        # scan and find best matching tile image
        final_errors = [tile.compare(source_region) for tile in self.tiles]

        if VERBOSE:
            min_err = min(final_errors)
            max_err = max(final_errors)
            self.min_error = min(min_err, self.min_error)
            self.max_error = max(max_err, self.max_error)

        # get the first index where error was equal to the smallest error
        # select the tile at that index
        best_idx = final_errors.index(min(final_errors))
        best_tile = self.tiles[best_idx]
        return self.add_subtle_overlay(best_tile.get_image(resize=(width, height)), source_region.img)

    def fit_tiles(self):
        """Fit all tiles in the Mosaic

        Scan through horizontal/vertical lines,
        matching a tile to each segment of the source image,
        and add it to the mosaic.
        """
        # if compare arrays have never been loaded, load the tiles.
        if not self.arrays_made:
            self.make_all_compare_arrays()

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
                cwrite(f'Comparing tile {tile_idx}/{total_tiles} ({tile_x}x{tile_y})')

                # find the tile(s) matching this x/y
                this_tile = self.find_tile(tile_x, tile_y, tile_width, tile_height)
                crop = (
                    tile_x * tile_width,
                    tile_y * tile_height,
                    tile_x * tile_width + tile_width,
                    tile_y * tile_height + tile_height,
                )
                self.mosaic.paste(
                    this_tile,
                    box=crop,
                )

        cprint(f'{total_tiles} tiles selected.', 'OKGREEN')

        if self.overlay_alpha:
            cprint('Applying overlay...', 'OKBLUE')
            self.mosaic = self.add_normal_overlay()

        if VERBOSE:
            cprint(f'Smallest error: {self.min_error}, largest error: {self.max_error}', color='UNDERLINE')

    def add_normal_overlay(self) -> Image.Image:
        """Overlay the source image over the whole mosaic"""
        overlay_img = ImageChops.overlay(self.mosaic, self.source_image)
        return ImageChops.blend(self.mosaic, overlay_img, self.overlay_alpha)

    def add_subtle_overlay(self, tile, source_tile) -> Image.Image:
        """Add one section of subtle overlay to a tile."""
        if not self.subtle_overlay_alpha:
            return tile
        return ImageChops.blend(
            tile,
            ImageChops.overlay(
                tile,
                source_tile.filter(ImageFilter.GaussianBlur(max(tile.width, tile.height) // 2)),
                ),
            self.subtle_overlay_alpha,
            )

    def save(self, show_preview, output_path):
        """Save (and/or show) the generated mosaic"""
        if show_preview:
            cprint('Showing img...', 'OKBLUE')
            self.mosaic.show()

        cprint('Saving img...', 'OKBLUE')
        self.mosaic.save(output_path)
        cprint(f'Saved as "{output_path}"', 'OKGREEN')


#                               __                      ,   ,
#                              (__` _|_  __╮ ,_ _|_     |\_/|  __╮ °  _
#                              .__)  |  (_/| |   |      |   | (_/| | | |
#                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        cprint(f'The program encountered an error: {e}', 'FAIL')
        input('\nPress Enter to quit.')
        raise
