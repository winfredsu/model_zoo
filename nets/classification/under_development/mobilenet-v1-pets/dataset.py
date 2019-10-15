import tensorflow as tf
import numpy as np
import os
import random
from config import *


AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_imgs(img_path):
    """
    args:
        img_path: directory containing all images
    return:
        splitted image paths
    """
    img_paths = os.listdir(img_path)
    img_paths.sort() # guarantee 'load_labels' gets the same paths
    img_paths = img_paths[:DATASET_SIZE]
    for i in range(len(img_paths)):
        img_paths[i] = os.path.join(img_path,img_paths[i])
    return img_paths
    # return img_paths[:TRAIN_SPLIT], img_paths[TRAIN_SPLIT:VAL_SPLIT], img_paths[VAL_SPLIT:]


def preprocess(img_path, label):
    img = tf.io.read_file(img_path)
    # uint8 range: [0,255]
    img = tf.image.decode_jpeg(img, channels=IMG_SHAPE[2])
    # new range: [-1.0,1.0)
    img = tf.image.resize(img, IMG_SHAPE[:2])
    img -= 128.0
    img /= 128.0
    return img, label

def get_dataset():
    """
    return:
        splitted dataset using tf.data API
    """
    # 1. get img_paths
    img_paths = os.listdir(IMG_PATH)
    img_paths.sort()
    img_paths = img_paths[:DATASET_SIZE]
    # 2. get all label names
    label_names = []
    for path in img_paths:
        cur_name = path.split('_')[0]
        if cur_name not in label_names:
            label_names.append(cur_name)
    # 3. get labels
    labels = []
    for i in range(len(img_paths)):
        labels.append(label_names.index(img_paths[i].split('_')[0]))
    labels = np.array(labels)
    # 4. get full image paths
    for i in range(len(img_paths)):
        img_paths[i] = os.path.join(IMG_PATH,img_paths[i])
    # 5. shuffle img_paths and labels with the same order
    randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(img_paths)
    random.seed(randnum)
    random.shuffle(labels)
    print(labels)
    # 6. create dataset
    img_paths_train, img_paths_val, img_paths_test = img_paths[:TRAIN_SPLIT], img_paths[TRAIN_SPLIT:VAL_SPLIT], img_paths[VAL_SPLIT:]
    labels_train, labels_val, labels_test = labels[:TRAIN_SPLIT], labels[TRAIN_SPLIT:VAL_SPLIT], labels[VAL_SPLIT:]

    ds_train = tf.data.Dataset.from_tensor_slices((img_paths_train, labels_train)).map(preprocess)
    ds_train = ds_train.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=4096))
    ds_train = ds_train.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    ds_val  = tf.data.Dataset.from_tensor_slices((img_paths_val, labels_val)).map(preprocess)
    ds_val  = ds_val.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=4096))
    ds_val  = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((img_paths_test, labels_test)).map(preprocess)
    ds_test = ds_test.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=4096))
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return ds_train, ds_val, ds_test
