# Data Acquiring

Download categories from [LSUN repo](http://dl.yf.io/lsun/objects/).

```bash
wget -c http://dl.yf.io/lsun/objects/[category] --tries=0 --read-timeout=20 --retry-connrefused --waitretry=1
```

Unarchive the database.

```bash
unzip [category_archive]
```

Extract images for each category.

```bash
python utils/data_db_reader.py export [category_directory] --out_dir [category_directory_extracted] --flat
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

Already existing files in the directory will be preserved. Any duplicated names will be renamed randomly. The labels file will be appended. Thus, categories can be merged one by one.

```bash
python utils/data_subset.py --source=[source_directory] --target=[target_directory] --selected=[number_of_images_per_category] --seed=[random_seed]
```

From the subset of data extract a subsubset for model tuning. An equal number of samples is randomly chosen from each category. The target directory will be recreated.

```bash
python utils/data_tuning.py --source=[source_directory] --target=[target_directory] --selected=[number_of_images_per_category] --seed=[random_seed]
```

<br/>
<br/>

# Model selection

```bash
python model_selection.py --model=[model] --img_dir=[image_directory] --labels_file=[labels_file_path] --seed=[random_seed]
```
