# The Oxford-IIIT Pets Dataset
A 37 category pet dataset with roughly 200 images for each class. The images have a large variations in scale, pose and lighting. All images have an associated ground truth annotation of breed, head ROI, and pixel level trimap segmentation.

This dataset can be used to evaluate the training scripts of classification/detection/segmentation models. But remember that for object detection or segmentation tasks, each sample in this dataset contains only one target object. 

# Download the Dataset
``` bash
# From model_zoo/datasets/Oxford-IIIT
mkdir raw_data
cd raw_data
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvf images.tar.gz
tar -xvf annotations.tar.gz
```

# Convert the dataset to desirable formats
## Create TFrecords for TensorFlow object detection API
Before creating the tfrecords, we should have TensorFlow object detection API already installed. 
``` bash
# From model_zoo/datasets/Oxford-IIIT
python `path to object_detection/dataset_tools/create_pet_tf_record.py` \
    --label_map_path=`path to object_detection/data/pet_label_map.pbtxt` \
    --data_dir=raw_data \
    --output_dir=.
```
These tfrecords can be used to train object detection models. 