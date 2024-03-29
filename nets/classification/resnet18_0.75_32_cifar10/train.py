#!/usr/bin/env python

import tensorflow as tf
import os
import sys
import model
import dataset
from config import *

# from tensorflow.contrib import quantize

tf.app.flags.DEFINE_string('train_dir', './train', 'training directory')
tf.app.flags.DEFINE_string('val_dir', './val', 'validation directory')
tf.app.flags.DEFINE_bool('quantize', False, 'enable quantization')
tf.app.flags.DEFINE_string('start_ckpt', '', 'ckpt from which to continue training (filename only)')
tf.app.flags.DEFINE_bool('is_first_finetuning', False, 'set True if finetuning from a float model for the first time. This flag resets global step.')
tf.app.flags.DEFINE_integer('train_step_max', 100000, 'max training steps')
tf.app.flags.DEFINE_integer('val_step_interval', 1000, 'how often to run validation')
tf.app.flags.DEFINE_integer('save_step_interval', 1000, 'how often to save ckpt')
tf.app.flags.DEFINE_float('lr_start', 0.045, 'start learning rate')
tf.app.flags.DEFINE_float('lr_finetune', 0.001, 'finetune learning rate')
FLAGS = tf.app.flags.FLAGS

def train_val_loop(ds_train=None, ds_val=None):

    """
    create a train-validate loop

    params:
        ds_train: dataset for training, should be a batched tf.Dataset
        ds_val: dataset for validation, should be a batched tf.Dataset
    """
    
    LR_DECAY_FACTOR = 0.999
    LR_DECAY_STEPS = int(TRAIN_SIZE/BATCH_SIZE*1.5)
    if FLAGS.quantize:
        LR_START = FLAGS.lr_finetune
    else:
        LR_START = FLAGS.lr_start
    
    ## create train & val dir
    if os.path.exists(FLAGS.train_dir) == False:
        os.mkdir(FLAGS.train_dir)
    if os.path.exists(FLAGS.val_dir) == False:
        os.mkdir(FLAGS.val_dir)

    ## start a new session
    sess = tf.InteractiveSession()
    
    ## dataset iterator
    ds_train_iterator = ds_train.make_initializable_iterator()
    next_train_images, next_train_labels = ds_train_iterator.get_next()
    ds_train_iterator.initializer.run()
    
    ds_val_iterator = ds_val.make_initializable_iterator()
    next_val_images, next_val_labels = ds_val_iterator.get_next()
    ds_val_iterator.initializer.run()

    ## images/labels placeholder
    images = tf.placeholder(tf.float32, [BATCH_SIZE]+IMG_SHAPE, name='images')
    labels = tf.placeholder(tf.int64, [BATCH_SIZE], name='labels')
    
    ## build model
    ## TODO: change this to resnet-18
    logits = model.my_resnet(images, num_classes=NUM_CLASSES, depth_multiplier=DEPTH_MULTIPLIER, is_training=True)

    ## create train_op
    # define loss_op
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    # define acc_op
    correct_pred = tf.equal(labels, tf.argmax(logits,1))
    acc_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # create quantized training graph
    if FLAGS.quantize:
        tf.contrib.quantize.create_training_graph(quant_delay=0)
    # config learning rate
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(LR_START, global_step, LR_DECAY_STEPS, LR_DECAY_FACTOR, staircase=True)
    # create train_op (global step add 1 here)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
       train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op, global_step)
    
    ## create summary and merge
    tf.summary.scalar('loss', loss_op, collections=['train', 'val'])
    tf.summary.scalar('accuracy', acc_op, collections=['train', 'val'])
    tf.summary.scalar('learning_rate', learning_rate, collections=['train', 'val'])
    train_merged_summaries = tf.summary.merge_all('train')
    val_merged_summaries = tf.summary.merge_all('val')

    ## saver
    saver = tf.train.Saver(tf.global_variables())
    
    ## writer
    train_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    val_writer   = tf.summary.FileWriter(FLAGS.val_dir, sess.graph)

    ## initialize variables
    tf.global_variables_initializer().run()
        
    ## load checkpoint
    if FLAGS.start_ckpt:
        if FLAGS.is_first_finetuning:
            # first restore variables with ignore_missing_vars
            variables_to_restore = tf.contrib.slim.get_variables_to_restore()
            restore_fn = tf.contrib.slim.assign_from_checkpoint_fn(
                os.path.join(FLAGS.train_dir, FLAGS.start_ckpt),
                variables_to_restore,
                ignore_missing_vars=True)
            restore_fn(sess)
            # then, reset global step
            global_step_reset = tf.assign(global_step, 0)
            sess.run(global_step_reset)
        else:
            saver.restore(sess, os.path.join(FLAGS.train_dir, FLAGS.start_ckpt))
    
    start_step = global_step.eval()
    
    for train_step in range(start_step, FLAGS.train_step_max+1):
        # get current global_step
        curr_step = global_step.eval()
        
        # get data batch
        images_batch, labels_batch = sess.run([next_train_images, next_train_labels])

        # train
        _, train_loss, train_summary = sess.run(
            [
                train_op,
                loss_op,
                train_merged_summaries
            ],
            feed_dict={
                images: images_batch,
                labels: labels_batch
            }
        )
        train_writer.add_summary(train_summary, curr_step)
        print('Step: ', curr_step, 'loss=', '{:.9f}'.format(train_loss))

        # validation
        if (curr_step != 0 and curr_step % FLAGS.val_step_interval == 0):
            total_val_acc = 0
            for i in range(0, VAL_SIZE, BATCH_SIZE):
                images_batch, labels_batch = sess.run([next_val_images, next_val_labels])
                val_acc, val_summary = sess.run(
                    [
                        acc_op,
                        val_merged_summaries
                    ],
                    feed_dict={
                        images: images_batch,
                        labels: labels_batch
                    }
                )
                total_val_acc += val_acc * BATCH_SIZE / VAL_SIZE
            val_writer.add_summary(val_summary, curr_step)
            print('Step: ', curr_step, 'Validation Accuracy = ', total_val_acc)
        
        # save checkpoint periodically
        if (curr_step != 0 and curr_step % FLAGS.save_step_interval == 0):
            if FLAGS.quantize:
                ckpt_name = 'model.quant.ckpt'
            else:
                ckpt_name = 'model.ckpt'
            saver.save(sess, os.path.join(FLAGS.train_dir, ckpt_name), global_step=curr_step)
            print('Step: ', curr_step, 'Saving to ', FLAGS.train_dir)

def main(unused_arg):
    ds_train, ds_val, _ = dataset.get_dataset()
    train_val_loop(ds_train, ds_val)    

if __name__ == '__main__':
    tf.app.run(main)
