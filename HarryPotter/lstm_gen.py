from attrdict import AttrDict
from collections import namedtuple

import pdb

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import LSTMCell
from tensorflow.python.ops import rnn_cell, seq2seq

import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.embeddings import Embedding
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Flatten
from keras.layers.recurrent import LSTM

import argparse
import time
import os
#import tensorflow as tf

CharGenOpts = namedtuple('ReferitOpts', [
    'steps',
    'model',
    'rnn_dim',
    'vocab',
    'layers'
    'batch_size',
    'lr'
    ])

class CharGen(object):
    def __init__(self, opts, infer=False):
        self.opts = opts
        pdb.set_trace()
        if infer:
            opts.batch_size = 1
            opts.steps = 1
        if opts.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif opts.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif opts.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("Model type not supported: {}".format(opts.model))

        cell = cell_fn(opts.rnn_dim)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * opts.layers)
        self.input_data = tf.placeholder(tf.int32, [opts.batch_size, opts.steps])
        self.targets = tf.placeholder(tf.int32, [opts.batch_size, opts.steps])
        self.init_state = cell.zero_state(opts.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [opts.rnn_dim, opts.vocab])
            softmax_b = tf.get_variable("softmax_b", [opts.vocab])
            with tf.device("/gpu:0"):
                embedding = tf.get_variable("embedding", [opts.vocab, opts.rnn_dim])
                inputs = tf.split(1, opts.steps, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_i, [1]) for input_i in inputs]

        def loop(prev, _):
            prev = tf.matmul(softmax_w, prev) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.init_state, self.cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, opts.rnn_dim])

        self.logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([opts.batch_size * opts.steps])],
                opts.vocab)

        self.cost = tf.reduce_sum(loss / opts.batch_size  / opts.steps)
        self.final_state = last_state

        # Set up training with grads
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), opts.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, chars, vocab, num=200, prime='The', sampling=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        for char in prime[:-1]:
            x = np.zeros(1, 1)
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.init_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            n = np.sum(weights)
            return int(np.searchsorted(t, np.random.rand(1)*s))

        ret = prime
        char = prime[-1]

        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.init_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling == 0:
                sample = np.argmax(p)
            elif sampling == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred

        return ret
        

#def main():
#    print "Hello world"
#    return
#
#if __name__ == '__main__':
#    main()
