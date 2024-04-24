# Data Acquiring

Download categories from [LSUS repo](http://dl.yf.io/lsun/objects/).

```bash
wget -c http://dl.yf.io/lsun/objects/[category] --tries=0 --read-timeout=20 --retry-connrefused --waitretry=1
```

Extract images for each category

```bash
python utils/data_db_reader.py export [category_file] --out_dir [category_directory_extracted] --flat
```

Randomly pick N images from each category, label them and save in a new directory.<br/>
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

Everything the new directory will be deleted

```bash
python utils/data_subset.py --source=[source_directory] --target=[target_directory] --selected=[number_of_images_per_category] --seed=[random_seed]
```