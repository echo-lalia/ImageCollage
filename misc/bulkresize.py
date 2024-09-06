from PIL import Image
import argparse
import os



parser = argparse.ArgumentParser(
    prog='bulkresize',
    description='This tool can be used to quickly resize images from a folder to a target size.'
)
parser.add_argument(
    'img_folder',
    help="A path to a folder of images to resize in place."
    )
parser.add_argument(
    '-w', '--width', type=int,
    help='Images will be larger than this width.'
)
parser.add_argument(
    '-H', '--height', type=int,
    help='Images will be larger than this height.'
)
parser.add_argument(
    '-wh', '--width_height', type=int,
    help='Set both width and height.'
)
parser.add_argument(
    '-i', '--in_place',
    help='modify files in-place',
    action='store_true'
)
parser.add_argument(
    '-s', '--skip_sized',
    help='Dont copy over images that are already sized.',
    action='store_true'
)

args = parser.parse_args()

if args.width is None and args.height is None and args.width_height is None:
    raise ValueError("Width and height must be set.")
if args.width is None:
    args.width = args.width_height
if args.height is None:
    args.height = args.width_height


if args.in_place:
    out_path = args.img_folder
else:
    out_path = os.path.join(os.getcwd(), f"{os.path.basename(args.img_folder)}_{args.width}x{args.height}")
    os.makedirs(out_path, exist_ok=True)


# this is just used to make the loading pretty:
class Printer:
    max_line = ''
    load_chars = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
    char_idx = 0

    def next_char(self):
        self.char_idx = (self.char_idx + 1) % len(self.load_chars)
        return self.load_chars[self.char_idx]
Printer = Printer()


def progress_print(text):
    text = str(text)
    text = f"{Printer.next_char()} - {text}"
    if len(text) > len(Printer.max_line):
        Printer.max_line = ' ' * len(text)
    print(Printer.max_line, end='\r')
    print(text, end='\r')

count_resized = 0
count_failed = 0
count_already_sized = 0
total_idx = 0
total_image_count = len(list(os.scandir(args.img_folder)))
for img_file in os.scandir(args.img_folder):
    total_idx += 1
    progress_print(f"Resizing {total_idx}/{total_image_count} ({img_file.name})...")
    try:
        img = Image.open(img_file)
        # dont bother with images that are already too small
        if img.width > args.width and img.height > args.height:
            # resize smallest dimension to match requested dimension
            width_from_height = int((args.height / img.height) * img.width) 
            height_from_width = int((args.width / img.width) * img.height)

            if width_from_height < args.width:
                target_width = args.width
                target_height = height_from_width
            else:
                target_height = args.height
                target_width = width_from_height

            img = img.resize((target_width, target_height))
            img.save(
                os.path.join(out_path, img_file.name)
            )
            count_resized += 1

        else:
            if not args.in_place and not args.skip_sized:
                img.save(
                    os.path.join(out_path, img_file.name)
                )
            count_already_sized += 1
    
    except Exception as e:
        count_failed += 1
        print(f"Warning: {img_file.name} couldn't be resized: {e}")


# clear leftover line
print(Printer.max_line)
for count, txt in (
    (total_image_count, 'total files scanned.'),
    (count_resized, 'images resized.'),
    (count_failed, 'failed to resize.'),
    (count_already_sized, 'were already correct size.')
    ):
    if count:
        print(f"{count} {txt}")

