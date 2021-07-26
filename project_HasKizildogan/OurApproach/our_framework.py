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
import functools
import tensorflow as tf
from absl import logging
from lib import dataset_utils
from lib import ssl_utils
from third_party.vat import vat_utils

#As flags are extensively used, importing it from Abseil
from absl import flags
#To save some extra typing
FLAGS = flags.FLAGS
#To perform extra operations
import numpy as np

#Same SSL Framework is kept, but added extra seen with comments
class SSLFramework(object):
    def __init__(
            self,
            network,
            hps,
            inputs,
            labels,
            #To perform UASD
            indexes,
            hist_predictions,
            #To perform OOD filtering
            threshold,
            unlabel,
            make_train_tensors,
            consistency_model,
            zca_input_file_path=None,
    ):
        """Init the class.

        Args:
            network (callable): Function which builds the graph for the
                network.
            hps (tf.contrib.training.HParams): all the hparams
            images (tensor): training images
            labels (int): training labels
            make_train_tensors (bool): make the tensors needed for training?
            consistency_model (str): which consistency regularization model to
                use.
            zca_input_file_path (str): path to pre-computed ZCA statistics.

        Returns:
            Initialized object.

        Raises:
            ValueError: if consistency_model is not valid.
        """

        logging.info("Building model with HParams %s", hps)
        self.network = network
        self.hps = hps
        self.consistency_model = consistency_model
        self.global_step = tf.train.get_or_create_global_step()

        # We need to wrap these tensors in identity calls so that passing
        # in the test data through a feed_dict will work at eval time
        self.inputs = tf.identity(inputs)
        self.labels = tf.identity(labels)
        #Indexes and predictions were added to class above so creating a
        #tensor with the same shape and contents
        self.indexes = tf.identity(indexes)
        self.hist_predictions = tf.identity(hist_predictions)
        self.threshold = tf.identity(threshold)
        self.unlabel = tf.identity(unlabel)
        self.is_training = tf.placeholder(
            dtype=tf.bool, shape=[], name="is_training"
        )

        if zca_input_file_path:
            logging.info(
                "Normalizing images with stats from: %s", zca_input_file_path
            )
            self.processed_images = self.inputs

            # A two step process that does the same as the
            # cifar_unnormalized -> cifar10 conversion process

            # 1. "De-normalize" back into [0, 255]
            self.processed_images /= 2
            self.processed_images += 0.5
            self.processed_images *= 255.0

            # 2. Apply Global Contrast Normalization and ZCA normalization
            # based on some dataset statistics passed in as a hyperparameter.
            self.processed_images = dataset_utils.tf_gcn(self.processed_images)
            self.processed_images = dataset_utils.zca_normalize(
                self.processed_images, zca_input_file_path
            )
        else:
            self.processed_images = self.inputs

        # logits is always the clean network output, and is what is used to evaluate the model.
        #
        # logits_student is the output that we want to make more similar to logits_teacher.
        # Each SSL method has its own concept of what the student and teacher outputs are, but
        # logits is guaranteed to be equal to either logits_student or logits_teacher.
        
        
        #Omit comments above from Oliver as UASD doesn't use that aproach
        self.logits, self.probs = (
            self.prediction()
        )

        labeled_mask = tf.not_equal(-1, labels[0:FLAGS.batch_size])
        masked_logits = tf.boolean_mask(self.logits[0:FLAGS.batch_size], labeled_mask)
        masked_labels = tf.boolean_mask(self.labels[0:FLAGS.batch_size], labeled_mask)

        unlabeled_mask = tf.equal(-1, labels[0:FLAGS.batch_size])

        # Getting scores of confidence for all unlabelled data to prepare for OOD
        self.unlabel_conf = tf.boolean_mask(tf.reduce_max(self.hist_predictions, 1), self.unlabel)

        #Accumulating the predictions as mentioned in ReadMe and paper
        pseudo_pred = tf.gather(self.hist_predictions, self.indexes[0:FLAGS.batch_size])
        confidence = tf.reduce_max(pseudo_pred, 1)
        
        #Creating one-hot vector to approximate with pseudo-labeling on unlabelled data
        if FLAGS.hard_label:
            pseudo_pred = tf.one_hot(tf.argmax(pseudo_pred, 1), FLAGS.num_classes)
        
        #Entropy for the loss is defined on SSL_Utils folder by Oliver et al.
        #We apply max probabilty and entropy calculations for the equations given
        #in the algorithm.
        
        #For unlabeled part
        unlabeled_prob = tf.boolean_mask(pseudo_pred, unlabeled_mask)
        self.entropy = tf.reduce_mean(ssl_utils.entropy_from_probs(unlabeled_prob))
        self.max_prob = tf.reduce_mean(tf.reduce_max(unlabeled_prob, 1))
        
        #For labeled part
        labeled_prob = tf.boolean_mask(pseudo_pred, labeled_mask)
        self.entropy_l = tf.reduce_mean(ssl_utils.entropy_from_probs(labeled_prob))
        self.max_prob_l = tf.reduce_mean(tf.reduce_max(labeled_prob, 1))

        #Hard labels where threshold is around 0.9~0.95
        conf_mask = tf.greater_equal(confidence, threshold)
        
        #Preparation for calculation of the loss function
        if FLAGS.all:
            print('Without OOD Filtering')
            masked_logits_all = self.logits[0:FLAGS.batch_size]
        else:
            pseudo_pred = tf.boolean_mask(pseudo_pred, conf_mask)
            masked_logits_all = tf.boolean_mask(self.logits[0:FLAGS.batch_size], conf_mask)

        #Squeezing the tensor by means of summation 
        self.unlabel_num = tf.reduce_sum(tf.to_int64(conf_mask))
        
        # # Splitting labels
        # if FLAGS.labeled_classes_filter is not "" and FLAGS.label_offset:
        #     label_offset = int(min(FLAGS.labeled_classes_filter.split(',')))
        # else:
        #     label_offset = 0
        # masked_labels = masked_labels - label_offset
        
        
        #Calculating accuracy again using the squeeze function but this time
        #for the labeled data
        self.accuracy = tf.reduce_mean(
            tf.to_float(
                tf.equal(
                    tf.argmax(masked_logits, axis=-1),
                    tf.to_int64(masked_labels),
                )
            )
        )
        
        
        #Applying the loss function suggested in the paper that is
        #L=H(y_true)+w*f.H(q)
        self.labeled_loss = tf.losses.softmax_cross_entropy(
            logits=masked_logits, onehot_labels=tf.one_hot(masked_labels, FLAGS.num_classes),
            label_smoothing=FLAGS.smoothing * FLAGS.num_classes)

        # Soft labels where the threshold is 0.95
        if FLAGS.MSE:
            self.unlabeled_loss = tf.reduce_mean(tf.reduce_mean(
                tf.square(tf.nn.softmax(masked_logits_all) - pseudo_pred), -1))
        else:
            self.unlabeled_loss = tf.losses.softmax_cross_entropy(
                logits=masked_logits_all, onehot_labels=pseudo_pred,
                label_smoothing=FLAGS.smoothing * FLAGS.num_classes)

        self.global_step_init = tf.initialize_variables([self.global_step])

        if make_train_tensors:
            self.make_train_tensors()

    def make_train_tensors(self):
        """Makes tensors needed for training this model.

        These shouldn't be made when just doing evaluation, both because it
        wastes memory and because some of these depend on hyperparameters that
        are unavailable during evaluation.

        Raises:
            ValueError: If given invalid hparams.
        """

        self.lr = tf.train.exponential_decay(
            self.hps.initial_lr,
            self.global_step,
            self.hps.lr_decay_steps,
            self.hps.lr_decay_rate,
            staircase=True,
        )

        # Multiplier warm-up schedule from Appendix B.1 of
        # the Mean Teacher paper (https://arxiv.org/abs/1703.01780)
        # "The consistency cost coefficient and the learning rate were ramped up
        # from 0 to their maximum values, using a sigmoid-shaped function
        # e^{−5(1−x)^2}, where x in [0, 1]." # todo: check
        
        
        #Same application as the original paper, just added if else for
        # the suggested consistency as shared above.
        # 100000 is the warm-up step number, it is like a hyperparameter
        #so can be optimized
        
        
        #It is literally the same function: e^{−5(1−x)^2}
        #However when we apply mean-squared-error we have the multiplication for
        #multiplier with the max multiplier
        #If it passes to else then we don't apply max multiplier
        
        if FLAGS.MSE:
            cons_multiplier = tf.cond(
                self.global_step < 100000,
                lambda: tf.exp(
                    -5.0
                    * tf.square(
                        1.0
                        - tf.to_float(self.global_step)
                        / tf.to_float(100000)
                    )
                ),
                lambda: tf.constant(1.0),
            )
            self.cons_multiplier = cons_multiplier * self.hps.max_cons_multiplier
            
            
        
        else:
            cons_multiplier = tf.cond(
                self.global_step < self.hps.warmup_steps,
                lambda: tf.exp(
                    -5.0
                    * tf.square(
                        1.0
                        - tf.to_float(self.global_step)
                        / tf.to_float(self.hps.warmup_steps)
                    )
                ),
                lambda: tf.constant(1.0),
            )

            # cons_multiplier = tf.cond(
            #     self.global_step < self.hps.warmup_steps,
            #     lambda: tf.maximum(0., (tf.to_float(self.global_step) - 100000)/100000),
            #     lambda: tf.constant(1.0),
            # )
            self.cons_multiplier = cons_multiplier

        if FLAGS.MSE:
            self.total_loss = self.labeled_loss + self.unlabeled_loss * self.cons_multiplier
        else:
            self.total_loss = tf.where(self.global_step < 100000, self.labeled_loss,
                                       self.labeled_loss + self.unlabeled_loss * self.cons_multiplier)

        with tf.control_dependencies(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ):
            self.train_op = tf.train.AdamOptimizer(
                learning_rate=self.lr
            ).minimize(self.total_loss, global_step=self.global_step)

        self.scalars_to_log = {
            "cons_multiplier": self.cons_multiplier,
            "accuracy": self.accuracy,
            "labeled_loss": self.labeled_loss,
            "unlabeled_loss": self.unlabeled_loss,
            "total_loss": self.total_loss,
            "lr": self.lr,
            "unlabel_num": self.unlabel_num,
            "unlabel_conf": tf.reduce_mean(self.unlabel_conf),
            "threshold": self.threshold,
            "entropy": self.entropy,
            "max_prob": self.max_prob,
            "entropy_l": self.entropy_l,
            "max_prob_l": self.max_prob_l,
        }

        for k, v in self.scalars_to_log.items():
            tf.summary.scalar(k, v)

    def prediction(self):
        """Actually get the outputs of the neural network."""

        network_function = functools.partial(
            self.network, is_training=self.is_training, hps=self.hps
        )
        inputs = self.processed_images
        with tf.variable_scope("prediction"):
            output = network_function(inputs, update_batch_stats=True)
            # Convert to probabilities so that we can use pseudo-label
            # threshold
            probs = tf.nn.softmax(output)

        return output, probs


#In the original framework, below there exist application for consistency 
#models mentioned in the paper. However, as it can easily be run with
#Oliver's file, they are all removed not to bother with function output.