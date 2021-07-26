#!/usr/bin/python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
import os
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from absl import logging
from lib import data_provider
from lib import dataset_utils
from lib import tf_utils
from lib import hparams
from lib.ssl_framework import SSLFramework
from lib import networks

#These are just here to save history. Feel free to change if you want
#to save files with different approach
from scipy.io import savemat
import pdb
_SAVE_SPAN = 400
_PREFETCH_BUFFER_SIZE = 200

 
_PRINT_SPAN = 300
_CHECK_TRIAL_EARLY_STOP = 100


# Flags for model training
flags.DEFINE_string(
    "hparam_string", None, "String from which we parse hparams."
)
flags.DEFINE_string(
    "primary_dataset_name", "svhn", "Name of dataset containing primary data."
)
flags.DEFINE_string(
    "secondary_dataset_name",
    "",
    "Name of dataset containing secondary data. Defaults to primary dataset",
)
flags.DEFINE_integer("label_map_index", 0, "Index of the label map.")

#Number of labeled classes are selected arbitrarily. Feel free to play with it
flags.DEFINE_integer("num_classes", 6, "Number of the class labels.")
#Number of samples is 50000 because this is how original Cifar10 train set 
#is constructed
flags.DEFINE_integer("num_samples", 50000, "Number of the samples.")
#
flags.DEFINE_integer(
    "n_labeled", -1, "Number of labeled examples, or -1 for entire dataset."
)
# flags.DEFINE_float('smoothing', 0.001, 'The smoothing factor in each class label.')
flags.DEFINE_float('smoothing', 0.0, 'The smoothing factor in each class label.')
flags.DEFINE_integer(
    "training_length", 500000, "number of steps to train for."
)
flags.DEFINE_integer(
    "warmup_steps", 100000, "number of steps to train for."
)
flags.DEFINE_integer("batch_size", 100, "Size of the batch")
#Oliver et al had it as mean_teacher but if such consistency model is used,
#original train file should be used.
flags.DEFINE_string(
    "consistency_model", "ours", "Which consistency model to use."
)
flags.DEFINE_string(
    "zca_input_file_path",
    "",
    "Path to ZCA input statistics. '' means don't ZCA.",
)

flags.DEFINE_float(
    "unlabeled_data_random_fraction",
    1.0,
    "The fraction of unlabeled data to use during training.",
)
flags.DEFINE_string(
    "labeled_classes_filter",
    "",
    "Comma-delimited list of class numbers from labeled "
    "dataset to use during training. Defaults to all classes.",
)
flags.DEFINE_string(
    "unlabeled_classes_filter",
    "",
    "Comma-delimited list of class numbers from unlabeled "
    "dataset to use during training. Useful for labeled "
    "datasets being used as unlabeled data. Defaults to all "
    "classes.",
)

# As the progress is saved, defining the root directory to store the experiments
flags.DEFINE_string(
    "root_dir", None, "The overall dir in which we store experiments"
)
flags.mark_flag_as_required("root_dir")

#To differentiate experiments along others.
flags.DEFINE_string(
    "experiment_name", "default", "The name of this particular experiment"
)
#To create checkpoints to inspect the process
flags.DEFINE_string(
    "load_checkpoint",
    "",
    "Checkpoint file to start training from (e.g. "
    ".../model.ckpt-354615), or None for random init",
)
flags.DEFINE_string(
    "dataset_mode",
    "mix",
    "'labeled' - use only labeled data to train the model. "
    "'unlabeled' - use only unlabel data to train the model"
    "'mix' (default) -  use mixed data to train the model")

#As a starting point, the confidence threshold mentioned in equation 3
#is created with booleans where all we have is label offsets.
#This can be by-passed by using another approach
flags.DEFINE_boolean('label_offset', True, '')
flags.DEFINE_boolean('stop', False, '')
flags.DEFINE_boolean('all', False, '')
flags.DEFINE_boolean('hard_label', False, '')
flags.DEFINE_boolean('majority', False, '')
flags.DEFINE_boolean('MSE', False, '')
flags.DEFINE_float('threshold', 0.9, 'Confidence Threshold.')
FLAGS = flags.FLAGS


def train(hps, result_dir, tuner=None, trial_name=None):
    """Construct model and run main training loop."""
    # Write hyperparameters to text summary
    hparams_dict = hps.values()
    # Create a markdown table from hparams.
    header = "| Key | Value |\n| :--- | :--- |\n"
    keys = sorted(hparams_dict.keys())
    lines = ["| %s | %s |" % (key, str(hparams_dict[key])) for key in keys]
    hparams_table = header + "\n".join(lines) + "\n"

    hparam_summary = tf.summary.text(
        "hparams", tf.constant(hparams_table, name="hparams"), collections=[]
    )


    with tf.Session() as sess:
        writer = tf.summary.FileWriter(result_dir, graph=sess.graph)
        writer.add_summary(hparam_summary.eval())
        writer.close()

    # We need to be able to run on the normal dataset for debugging.
    if FLAGS.n_labeled != -1:
        label_map = "label_map_count_{}_index_{}".format(
            FLAGS.n_labeled, FLAGS.label_map_index
        )
    else:
        label_map = None

    container_name = trial_name or ""


    # Create a separate container for each run so parameters don't stick around
    with tf.container(container_name):

        if label_map:
            label_table = dataset_utils.construct_label_table(
                FLAGS.primary_dataset_name, label_map
            )
        else:
            label_table = None

        labeled_data_filter_fn = make_labeled_data_filter_fn(label_table)
        unlabeled_data_filter_fn = make_unlabeled_data_filter_fn()

        # Accumulating predictions
        predictions_acc = np.full((FLAGS.num_samples, FLAGS.num_classes), 0.0)
        valid_mask = np.full((FLAGS.num_samples), 0.0)
        unlabel_mask = np.full((FLAGS.num_samples), 0.0)
        valid_acc_mask = np.full((FLAGS.num_samples), 0.0)
        th_valid = 0.9

        images, labels, indexes, _, _, _, _ = data_provider.get_simple_mixed_batch(
            labeled_dataset_name=FLAGS.primary_dataset_name,
            unlabeled_dataset_name=(
                FLAGS.secondary_dataset_name or
                FLAGS.primary_dataset_name),
            split="train",
            batch_size=FLAGS.batch_size,
            shuffle_buffer_size=1000,
            labeled_data_filter_fn=labeled_data_filter_fn,
            unlabeled_data_filter_fn=unlabeled_data_filter_fn,
            mode=FLAGS.dataset_mode,
        )
        
        #Creating validation part same as above for the unlabeled part.
        images_v, labels_v, indexes_v, _, _, _, _ = data_provider.get_simple_mixed_batch(
            labeled_dataset_name=FLAGS.primary_dataset_name,
            unlabeled_dataset_name=(
                FLAGS.secondary_dataset_name or
                FLAGS.primary_dataset_name),
            split="valid",
            batch_size=int(FLAGS.batch_size*0.1),
            shuffle_buffer_size=100,
            labeled_data_filter_fn=labeled_data_filter_fn,
            unlabeled_data_filter_fn=None,
            mode="unlabeled",
        )

        images_v, labels_v, indexes_v = make_images_and_labels_tensors(-1)
        
        #Concatenating to work with train and validation
        images = tf.concat([images, images_v], 0)
        labels = tf.concat([labels, labels_v], 0)
        indexes = tf.concat([indexes, indexes_v], 0)
        
        #This line won't work on TF 2 as logging is dropped.
        logging.info("Training data tensors constructed.")
        # This is necessary because presently svhn data comes as uint8
        images = tf.cast(images, tf.float32)

        # Accumulated historical predictions
        hist_predictions = tf.placeholder(tf.float32, shape=[FLAGS.num_samples, FLAGS.num_classes])
        threshold = tf.placeholder(tf.float32, shape=[])
        unlabel = tf.placeholder(tf.bool, shape=[FLAGS.num_samples])


        #Same WideResNet architecture modified with new framework.
        ssl_framework = SSLFramework(
            networks.wide_resnet,
            hps,
            images,
            labels,
            indexes,
            hist_predictions,
            threshold,
            unlabel,
            make_train_tensors=True,
            consistency_model=FLAGS.consistency_model,
            zca_input_file_path=FLAGS.zca_input_file_path,
        )
        tf.summary.scalar("n_labeled", FLAGS.n_labeled)
        tf.summary.scalar("batch_size", FLAGS.batch_size)

        logging.info("Model instantiated.")
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(max_to_keep=1)

        if FLAGS.load_checkpoint:
            vars_to_load = [
                v for v in tf.all_variables() if "logit" not in v.name
            ]
            finetuning_saver = tf.train.Saver(
                keep_checkpoint_every_n_hours=1, var_list=vars_to_load
            )

        def init_fn(_, sess):
            sess.run(init_op)
            if FLAGS.load_checkpoint:
                logging.info(
                    "Fine tuning from checkpoint: %s", FLAGS.load_checkpoint
                )
                finetuning_saver.restore(sess, FLAGS.load_checkpoint)

        scaffold = tf.train.Scaffold(
            saver=saver, init_op=ssl_framework.global_step_init, init_fn=init_fn
        )
        
        #Original saves a checkpoint every 5 mins, however feel free to replace
        #the whole part below up to line 323 for other approach
        
        logging.info("Scaffold created.")
        monitored_sess = tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            checkpoint_dir=result_dir,
            save_summaries_secs=10,
            save_summaries_steps=None,
            config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False
            ),
            max_wait_secs=300,
            save_checkpoint_steps=500,
        )
        logging.info("MonitoredTrainingSession initialized.")
        trainable_params = np.sum(
            [
                np.prod(v.get_shape().as_list())
                for v in tf.trainable_variables()
            ]
        )
        logging.info("Trainable parameters: %s", str(trainable_params))

        def should_stop_early():
            if tuner and tuner.should_trial_stop():
                logging.info(
                    "Got tuner.should_trial_stop(). Stopping trial early."
                )
                return True
            else:
                return False

        #Storing histogram data to the rood directory
        count = 0
        dirName = os.path.join(FLAGS.root_dir, 'histogram')
        if not os.path.exists(dirName):
            os.makedirs(dirName)

        hist_valid = None
        hist_unlabel = None

        with monitored_sess as sess:

            while True:
                #Normalizing predictions under the boundary of 1
                feed_pred = predictions_acc / np.maximum(np.sum(predictions_acc, 1, keepdims=True), 1.0)

                _, logits, probs, indexes, labels, unlabel_num, step, accuracy, values_to_log = sess.run(
                    [
                        ssl_framework.train_op,
                        ssl_framework.logits,
                        ssl_framework.probs,
                        ssl_framework.indexes,
                        ssl_framework.labels,
                        ssl_framework.unlabel_num,
                        ssl_framework.global_step,
                        ssl_framework.accuracy,
                        ssl_framework.scalars_to_log,
                    ],
                    feed_dict={ssl_framework.is_training: True,
                               hist_predictions: feed_pred,
                               unlabel: np.array(unlabel_mask, dtype=bool),
                               threshold: th_valid},
                )

                # Calculating accuracy with argmax function like
                #in the lecture
                valid_acc = (labels[FLAGS.batch_size:] == np.argmax(probs[FLAGS.batch_size:, :], axis=1))

                if step % _SAVE_SPAN == 0 and (hist_valid is not None) and (hist_unlabel is not None):
                    savefile = os.path.join(FLAGS.root_dir, 'predictions_acc.mat')
                    savemat(savefile, {"predictions_acc": predictions_acc})

                    savefile = os.path.join(FLAGS.root_dir, 'histogram', 'hist_valid_' + str(count) + '.mat')
                    savemat(savefile, {"hist_valid": hist_valid})

                    savefile = os.path.join(FLAGS.root_dir, 'histogram', 'hist_unlabel_' + str(count) + '.mat')
                    savemat(savefile, {"hist_unlabel": hist_unlabel})

                if step >= FLAGS.training_length:
                    break
                # Don't call should_stop_early() too frequently
                if step % _CHECK_TRIAL_EARLY_STOP == 0 and should_stop_early():
                    break


def make_images_and_labels_tensors(examples_to_take):
    """Make tensors for loading images and labels from dataset."""

    with tf.name_scope("input"):
        dataset = dataset_utils.get_dataset(
            FLAGS.primary_dataset_name, 'valid' #FLAGS.split
        )
        dataset = dataset.filter(make_labeled_data_filter())

        # be repeated indefinitely.
        dataset = dataset.cache().repeat()

        # This is necessary for datasets that aren't shuffled on disk, such as
        # ImageNet.
        # if FLAGS.split == "train":
        #     dataset = dataset.shuffle(FLAGS.shuffle_buffer_size, 0)

        # Optionally only use a certain fraction of the dataset.
        # This is used in at least 2 contexts:
        # 1. We don't evaluate on all training data sometimes for speed reasons.
        # 2. We may want smaller validation sets to see whether HPO still works.
        if examples_to_take != -1:
            dataset = dataset.take(examples_to_take)

        # Batch the results: 10% data is validation set
        dataset = dataset.batch(int(FLAGS.batch_size*0.1))
        dataset = dataset.prefetch(_PREFETCH_BUFFER_SIZE)

        # Get the actual results from the iterator
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(
            tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer
        )
        images, labels, indexes, _ = iterator.get_next()
        images = tf.cast(images, tf.float32)

    return images, labels, indexes


def make_labeled_data_filter():
    """Make filter for certain classes of labeled data."""
    if FLAGS.primary_dataset_name in {"cifar100", "tinyimagenet_32", "cifar100_tinyimagenet"}:
        labels = FLAGS.labeled_classes_filter.split(',')
        labels = range(int(labels[0]), int(labels[1]))
        labeled_classes_filter = ",".join([str(x) for x in labels])
    else:
        labeled_classes_filter = FLAGS.labeled_classes_filter
    print(labeled_classes_filter)
    class_filter = tf_utils.filter_fn_from_comma_delimited(
        labeled_classes_filter
    )
    return lambda image, label, index, fkey: class_filter(label)


def make_labeled_data_filter_fn(label_table):
    """Make filter for certain classes of labeled data."""
    if FLAGS.primary_dataset_name in {"cifar100", "tinyimagenet_32", "cifar100_tinyimagenet"}:
        labels = FLAGS.labeled_classes_filter.split(',')
        labels = range(int(labels[0]), int(labels[1]))
        labeled_classes_filter = ",".join([str(x) for x in labels])
    else:
        labeled_classes_filter = FLAGS.labeled_classes_filter
    print(labeled_classes_filter)
    class_filter = tf_utils.filter_fn_from_comma_delimited(
        labeled_classes_filter
    )
    if label_table:
        return lambda _, label, index, fkey: class_filter(label) & label_table.lookup(
            fkey
        )
    else:
        return lambda _, label, index, fkey: class_filter(label)


def make_unlabeled_data_filter_fn():
    """Make filter for certain classes and a random fraction of unlabeled
    data."""
    if FLAGS.secondary_dataset_name is "cifar100":
        labels = FLAGS.unlabeled_classes_filter.split(',')
        labels = range(int(labels[0]), int(labels[1]))
        labeled_classes_filter = ",".join([str(x) for x in labels])
    else:
        labeled_classes_filter = FLAGS.unlabeled_classes_filter
    print(labeled_classes_filter)
    class_filter = tf_utils.filter_fn_from_comma_delimited(
        labeled_classes_filter
    )

    def random_frac_filter(fkey):
        return tf_utils.hash_float(fkey) < FLAGS.unlabeled_data_random_fraction

    return lambda _, label, index, fkey: class_filter(label) & random_frac_filter(
        fkey
    )


def main(_):
    result_dir = os.path.join(FLAGS.root_dir, FLAGS.experiment_name)
    hps = hparams.get_hparams(
        FLAGS.primary_dataset_name, FLAGS.consistency_model
    )
    if FLAGS.hparam_string:
        hps.parse(FLAGS.hparam_string)
    train(hps, result_dir)
    if FLAGS.stop:
        pdb.set_trace()

if __name__ == "__main__":
    app.run(main)
