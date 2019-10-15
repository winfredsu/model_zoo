#!/usr/bin/env python

import tensorflow as tf
import os
import sys

from model import yolov3
from config import *
import os

import tensorflow as tf
import numpy as np
import logging
from tqdm import trange
import quantize
import args

from utils.data_utils import get_batch_data
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import gpu_nms
#sys.path.append('../..')
import quantize

tf.app.flags.DEFINE_string('train_dir', './train', 'training directory')

tf.app.flags.DEFINE_string('ckpt', 'model.quant.ckpt', 'ckpt to be frozen')
tf.app.flags.DEFINE_string('output_node', 'MobilenetV1/Squeeze', 'name of output name')
tf.app.flags.DEFINE_string('frozen_pb_name', 'frozen.pb', 'output pb name')
FLAGS = tf.app.flags.FLAGS

def freeze():
    sess = tf.InteractiveSession()
    # setting placeholders
    is_training = tf.placeholder(tf.bool, name="phase_train")
    handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
    # register the gpu nms operation here for the following evaluation scheme
    pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
    pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])


    ##################
    # tf.data pipeline
    ##################
    train_dataset = tf.data.TextLineDataset(args.train_file)
    train_dataset = train_dataset.shuffle(args.train_img_cnt)
    train_dataset = train_dataset.batch(args.batch_size)
    train_dataset = train_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, args.class_num, args.img_size, args.anchors, 'train', args.multi_scale_train,
                                  args.use_mix_up, args.letterbox_resize],
                             Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    train_dataset = train_dataset.prefetch(args.prefetech_buffer)

    val_dataset = tf.data.TextLineDataset(args.val_file)
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.map(
        lambda x: tf.py_func(get_batch_data,
                             inp=[x, args.class_num, args.img_size, args.anchors, 'val', False, False,
                                  args.letterbox_resize],
                             Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
        num_parallel_calls=args.num_threads
    )
    val_dataset.prefetch(args.prefetech_buffer)

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iterator.make_initializer(train_dataset)
    val_init_op = iterator.make_initializer(val_dataset)

    # get an element from the chosen dataset iterator
    image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
    y_true = [y_true_13, y_true_26, y_true_52]

    # tf.data pipeline will lose the data `static` shape, so we need to set it manually
    image_ids.set_shape([None])
    image.set_shape([None, None, None, 3])
    for y in y_true:
        y.set_shape([None, None, None, None, None])

    ##################
    # Model definition
    ##################
    yolo_model = yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay,
                        args.weight_decay, use_static_shape=False)

    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(image, is_training=is_training)

    quantize.create_eval_graph()


    # write frozen graph
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, os.path.join(FLAGS.train_dir, FLAGS.ckpt))

    frozen_gd = tf.graph_util.convert_variables_to_constants(
        sess, sess.graph_def, [FLAGS.output_node])
    tf.train.write_graph(
        frozen_gd,
        '/home/shihaobing/',
        FLAGS.frozen_pb_name,
        as_text=False)

def main(unused_arg):
    freeze()

if __name__ == '__main__':
    tf.app.run(main)
