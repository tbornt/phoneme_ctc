from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops

from phoeme_set import phoeme_set_39
from utils import process_wav


num_features = 39 # 12 mfcc + 26 logfbank
num_classes = 40 # 39 phonemes + blank

num_layers = 3 # 3 lstm cells stack together
num_hidden = 128 # lstm state

num_epochs = 140 
batch_size = 128

learning_rate = 0.001
momentum = 0.9


def decode_wav(ENV):
    graph = tf.Graph()
    with graph.as_default():
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_stepsize, num_features], but the
        # batch_size and max_stepsize can vary along each step
        inputs = tf.placeholder(tf.float32, [batch_size, None, num_features])

        targets_idx = tf.placeholder(tf.int64)
        targets_val = tf.placeholder(tf.int32)
        targets_shape = tf.placeholder(tf.int64)
        targets = tf.SparseTensor(targets_idx, targets_val, targets_shape)
        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [batch_size])

        # Weights & biases
        weight_classes = tf.Variable(tf.truncated_normal([2*num_hidden, num_classes],
                                                         stddev=np.sqrt(2.0/(2*num_hidden))))
        bias_classes = tf.Variable(tf.zeros([num_classes]))

        # Network
        forward_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True)
        backward_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, state_is_tuple=True)

        stack_forward_cell = tf.nn.rnn_cell.MultiRNNCell([forward_cell] * num_layers,
                                                         state_is_tuple=True)
        stack_backward_cell = tf.nn.rnn_cell.MultiRNNCell([backward_cell] * num_layers,
                                                          state_is_tuple=True)

        """
        # Reshaping to (step_size*batch_size, num_features)
        inputs_rs = tf.reshape(inputs, [-1, num_features])
        inputs_list = tf.split(0, ENV.max_step_size, inputs_rs)
        """

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(stack_forward_cell, 
                                                     stack_backward_cell,
                                                     inputs,
                                                     sequence_length=seq_len,
                                                     time_major=False, # [batch_size, max_time, num_hidden]
                                                     dtype=tf.float32)
        inputs_shape = tf.shape(inputs)

        outputs_concate = tf.concat_v2(outputs, 2)
        outputs_concate = tf.reshape(outputs_concate, [-1, 2*num_hidden]) # batch_size x step_size x num_hidden

        logits = tf.matmul(outputs_concate, weight_classes) + bias_classes
        logits = tf.reshape(logits, [batch_size, -1, num_classes])
        loss = tf.reduce_mean(ctc_ops.ctc_loss(logits, targets, seq_len, time_major=False))
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

        # Evaluating
        decoded, log_prob = ctc_ops.ctc_greedy_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_len)
        label_error_rate = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    config = tf.ConfigProto(device_count = {'GPU': 0})

    with tf.Session(config=config, graph=graph) as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        ckpt = tf.train.get_checkpoint_state(ENV.model_path)
        print('load', ckpt.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
        
        while True:
            wav_file = raw_input('Enter the path for a wav file:')
            print(wav_file)
            features = process_wav(wav_file)
            batch_features = np.array([features for i in range(128)])
            batch_seq_len = np.array([features.shape[0] for i in range(128)])
            print(batch_features.shape)
            feed = {
                inputs: batch_features,
                seq_len: batch_seq_len
            }
            d, oc = sess.run([decoded[0], outputs], feed_dict=feed)
            print(oc[0].shape)
            print(oc[1].shape)
            dsp = d.shape
            res = []
            for label in d.values[:dsp[1]]:
                for k, v in phoeme_set_39.items():
                    if v == label + 1:
                        res.append(k)           
            print(res)
