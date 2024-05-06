import argparse
import os
import shutil
import random
from shared import load_labels, write_labels, IMAGE_SUBFOLDER, LABELS_FILENAME

__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


def pick(source_folder, destination_folder, limit_per_category):
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    os.makedirs(destination_folder)
    os.makedirs(os.path.join(destination_folder, IMAGE_SUBFOLDER))
    labels = load_labels(source_folder, LABELS_FILENAME)
    if limit_per_category > min([list(labels.values()).count(cat) for cat in set(labels.values())]):
        raise Exception(f'Some categories contain less than {limit_per_category} labels')
    labels = {file: cat for cat in set(labels.values()) for file in random.sample([file for file, label_cat in labels.items() if label_cat == cat], limit_per_category)}
    for file, _ in labels.items():
        source_path = os.path.join(source_folder, IMAGE_SUBFOLDER, file)
        destination_path = os.path.join(destination_folder, IMAGE_SUBFOLDER, file)
        shutil.copy(source_path, destination_path)
    write_labels(destination_folder, LABELS_FILENAME, labels)

def count(labels):
    category_file_count = {}
    for category in set(labels.values()):
        count = sum(1 for file_cat in labels.values() if file_cat == category)
        category_file_count[category] = count
    return category_file_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='')
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--selected', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    pick(args.source, args.target, args.selected)

if __name__ == '__main__':
    main()