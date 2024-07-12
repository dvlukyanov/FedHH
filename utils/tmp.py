from shared import load_labels, write_labels, LABELS_FILENAME, IMAGE_SUBFOLDER

# labels = load_labels('data/tuning', 'labels.csv')
# print(set(labels.values()))

import os
from PIL import Image
from tqdm import tqdm

def check_images(directory):
    labels = load_labels(directory, 'labels.csv')
    print(f'Labels before: {len(labels)}')
    images_directory = os.path.join(directory, IMAGE_SUBFOLDER)
    bad_images = []
    for filename in tqdm(os.listdir(images_directory)):
        file_path = os.path.join(images_directory, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()
        except (IOError) as e:
            print(f'{file_path} is invalid')
            bad_images.append(filename)
    for bad_image in bad_images:
        if bad_image in labels:
            del labels[bad_image]
        os.remove(os.path.join(images_directory, bad_image))
        print(f'{bad_image} was removed')
    print(f'Labels after: {len(labels)}')
    print(f'Files removed: {len(bad_images)}')
    write_labels(directory, LABELS_FILENAME, labels)

directory_path = '../data/tuning'
check_images(directory_path)