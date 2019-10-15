# RSOD Dataset
This is an dataset for object detection in google earth images. The datset includes aircraft, oiltank, playground and overpass. 

## Create TFrecords for TensorFlow Object Detection API
- Download the datasets (here)[https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-]
- Uncompress all four datasets in `raw_data/dataset-original`. After this, you should get four directories `aircraft`, `oiltank`, `overpass` and `playground` in `raw_data/dataset-original`
- Put all images into `raw_data/JPEGImages` folder.
- In `raw_data` directory, execute `python ../utils/split_dataset.py`. This step copies the annotation files of selected class(es) to `raw_data/Annotations`, and split them into train and val sets. Modify this file to change the classes to be detected. (`utils/RSOD_label_map.pbtxt` should be also modified.) 
- In `raw_data` directory, execute `python ../utils/pascal2tfrecords.py` to create tfrecords for train set. And execute `python ../utils/pascal2tfrecords.py --annotations_dir Annotation/val --output_path ../tfrecord/RSOD_val.record` to create tfrecords for val set. 



