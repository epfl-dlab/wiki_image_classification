import cairosvg
import os
from PIL import Image

def svg_or_gif_to_png(svg_or_gif_path, output_directory):
    base_filename = os.path.basename(svg_or_gif_path)[:-4] # only get the filename without the ending .svg
    file_extension = os.path.splitext(svg_or_gif_path)[1].lower()
    converted_path = output_directory + base_filename + '.png'

    if os.path.exists(converted_path):
        return converted_path

    if file_extension == '.svg':
        try:
            print('try svg2png: ' + svg_or_gif_path)
            cairosvg.svg2png(url=svg_or_gif_path, write_to=converted_path)
            return converted_path
        except Exception as e:
            import traceback
            traceback.print_exc()
            return
    elif file_extension == '.gif':
        try:
            with Image.open(svg_or_gif_path) as image:
                image.save(converted_path, 'PNG')
            return converted_path
        except Exception as e:
            return
    return 

def convert_to_png(labels):
    print('Starting conversion...')

    output_directory = '/scratch/WIT_Dataset/converted_images/'
    counter = 0
    for index, row in labels.iterrows():
        image_path = row['url']
        # This image gives a segmentation fault when running cairosvg.svg2png
        if image_path == '/scratch/WIT_Dataset/images/b/b5/Splicing_by_Overlap_Extension_PCR.svg':
        #    image_path == '/scratch/WIT_Dataset/images/2/27/ChIP-on-chip_workflow_overview.svg':
            continue
        counter += 1
        if counter % 10_000 == 1:
            print(f'\n\n\n\ncounter: {counter}\n\n\n\n')
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension in ['.gif', '.svg']:
            if file_extension == '.svg' and index > 1_900_000:
                continue
            converted_path = svg_or_gif_to_png(image_path, output_directory)
            if converted_path:
                labels.loc[index, 'url'] = converted_path

    print('DONE with conversion!')
    return labels