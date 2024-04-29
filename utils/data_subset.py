import argparse
import os
import shutil
import random
import string
import csv

__author__ = 'Dmitry Lukyanov'
__email__ = 'dlukyan@clemson.edu'
__license__ = 'MIT'


IMAGE_SUBFOLDER = '/images'
LABELS_FILENAME = 'labels.csv'


def merge(source_folder, destination_folder, limit_per_category):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        os.makedirs(destination_folder + IMAGE_SUBFOLDER)
    labels = load_labels(destination_folder)
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
            labels.update({image: category for image in selected_images})
            for image in selected_images:
                image_path = os.path.join(category_folder, image)
                destination_path = os.path.join(destination_folder + IMAGE_SUBFOLDER, image)
                while os.path.exists(destination_path):
                    filename = generate_name() + '.' + image.split('.')[1]
                    destination_path = os.path.join(destination_folder+ IMAGE_SUBFOLDER, filename)
                    labels[image] = category
                    print(f'Already exists: {image}. Renamed to {filename}')
                shutil.copy(image_path, destination_path)
                cnt += 1
                if cnt % 1000 == 0:
                    print(f'{cnt} images are merged')
    write_labels(destination_folder, labels)


def load_labels(destination_folder):
    labels = {}
    csv_file = os.path.join(destination_folder, LABELS_FILENAME)
    if os.path.exists(csv_file) and os.path.isfile(csv_file):
        with open(csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            labels = {rows[0]: rows[1] for rows in reader}
    return labels


def generate_name(length=32):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def write_labels(destination_folder, labels):
    csv_file = os.path.join(destination_folder, LABELS_FILENAME)
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        for image, label in labels.items():
            writer.writerow([image, label])


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