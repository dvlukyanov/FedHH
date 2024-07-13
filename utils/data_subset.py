import argparse
import os
import shutil
import random
import string
from shared import load_labels, write_labels, IMAGE_SUBFOLDER, LABELS_FILENAME
from tqdm import tqdm
from PIL import Image


__author__ = 'Dmitry Lukyanov, Huaye Li'
__email__ = 'dmitry@dmitrylukyanov.com, huayel@g.clemson.edu'
__license__ = 'MIT'


def merge(source_folder, destination_folder, limit_per_category, seed):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    if not os.path.exists(os.path.join(destination_folder, IMAGE_SUBFOLDER)):
        os.makedirs(os.path.join(destination_folder, IMAGE_SUBFOLDER))
    labels = load_labels(destination_folder, LABELS_FILENAME)
    categories = [folder for folder in os.listdir(source_folder) if not folder.startswith('.')]
    for category in categories:
        print(f'Copying: {category}')
        cnt = 0
        category_folder = os.path.join(source_folder, category)
        if os.path.isdir(category_folder):
            images = os.listdir(category_folder)
            print('Filtering invalid images...')
            images = [image for image in tqdm(images) if is_valid(os.path.join(category_folder, image))]
            if len(images) < limit_per_category:
                raise Exception(f'Not enough images in {category}')
            random.seed(seed)
            selected_images = random.sample(images, limit_per_category)
            for image in selected_images:
                filename = image
                image_path = os.path.join(category_folder, filename)
                destination_path = os.path.join(destination_folder, IMAGE_SUBFOLDER, filename)
                while os.path.exists(destination_path):
                    filename = generate_name() + '.' + filename.split('.')[1]
                    destination_path = os.path.join(destination_folder, IMAGE_SUBFOLDER, filename)
                    print(f'Already exists. Renamed to {filename}')
                labels[filename] = category
                shutil.copy(image_path, destination_path)
                cnt += 1
                if cnt % 1000 == 0:
                    print(f'{cnt} images are merged')
    write_labels(destination_folder, LABELS_FILENAME, labels)


def is_valid(image):
    try:
        with Image.open(image) as img:
            img.verify()
        return True
    except (IOError) as e:
        print(f'{image} is invalid and will be skipped')
        return False


def generate_name(length=40):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--selected', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    merge(args.source, args.target, args.selected, args.seed)


if __name__ == '__main__':
    main()