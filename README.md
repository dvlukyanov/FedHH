<!--
__author__ = 'Dmitry Lukyanov, Huaye Li'
__email__ = 'dmitry@dmitrylukyanov.com, huayel@g.clemson.edu'
__license__ = 'MIT'
-->


# Data Acquiring

<br/>
Download categories from [LSUN repo](http://dl.yf.io/lsun/objects/):

```bash
wget -c http://dl.yf.io/lsun/objects/[category] --tries=0 --read-timeout=20 --retry-connrefused --waitretry=1
```
for example
```bash
wget -c http://dl.yf.io/lsun/objects/sheep.zip --tries=0 --read-timeout=20 --retry-connrefused --waitretry=1
```

<br/>
Unarchive the database:

```bash
unzip [category_archive]
```
for example, in data/original
```bash
unzip sheep.zip
```

Extract images for each category:

```bash
python code/utils/data_db_reader.py export [category_directory] --out_dir [category_directory_extracted] --flat
```
for example
```bash
python code/utils/data_db_reader.py export data/original/sheep --out_dir data/extracted/sheep --flat
```

Randomly pick N images from each category, label them and save into a new directory.<br/>
Each original category directory should be within a common one. E.g.,<br/>

data<br/>
&emsp;[category_1]<br/>
&emsp;&emsp;[images_from_category_1]<br/>
&emsp;[category_2]<br/>
&emsp;&emsp;[images_from_category_3]<br/>
&emsp;etc<br/>

will become<br/>

data_subset<br/>
&emsp;images<br/>
&emsp;&emsp;[all images]<br/>
&emsp;labels.csv<br/>

Already existing files in the directory will be preserved. Any duplicated names will be renamed randomly. The labels file will be appended. Thus, categories can be merged one by one, but if the subset operation has been executed on a category, its presence in the source directory will lead to duplications in the target directory

```bash
python utils/data_subset.py --source=[source_directory] --target=[target_directory] --selected=[number_of_images_per_category] --seed=[random_seed]
```
for example
```bash
python utils/data_subset.py --source=data/extracted --target=data/subset --selected=100000 --seed=0
```

From the subset of data extract a subsubset for model tuning. An equal number of samples is randomly chosen from each category. The target directory will be recreated.

```bash
python utils/data_tuning.py --source=[source_directory] --target=[target_directory] --selected=[number_of_images_per_category] --seed=[random_seed]
```
for example
```bash
python utils/data_tuning.py --source=data/subset --target=data/tuning --selected=1000 --seed=0
```

<br/>
<br/>

# Model selection

```bash
python model_selection.py --model=[model] --img_dir=[image_directory] --labels_file=[labels_file_path] --seed=[random_seed]
```
