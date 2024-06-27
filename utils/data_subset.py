import argparse
import os
import shutil
import random
import string
from shared import load_labels, write_labels, IMAGE_SUBFOLDER, LABELS_FILENAME


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


def merge(source_folder, destination_folder, limit_per_category):
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
            if len(images) < limit_per_category:
                raise Exception(f'Not enough images in {category}')
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


def generate_name(length=40):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--selected', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    merge(args.source, args.target, args.selected)


if __name__ == '__main__':
    main()