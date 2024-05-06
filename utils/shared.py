import os
import csv

__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


IMAGE_SUBFOLDER = 'images'
LABELS_FILENAME = 'labels.csv'


def load_labels(folder, filename):
    labels = {}
    csv_file = os.path.join(folder, filename)
    if os.path.exists(csv_file) and os.path.isfile(csv_file):
        with open(csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            labels = {rows[0]: rows[1] for rows in reader}
    return labels


def write_labels(folder, filename, labels):
    csv_file = os.path.join(folder, filename)
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label'])
        for image, label in labels.items():
            writer.writerow([image, label])