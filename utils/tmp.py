from shared import load_labels

labels = load_labels('data/tuning', 'labels.csv')
print(set(labels.values()))