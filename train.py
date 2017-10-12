import tensorflow as tf
import numpy as np
import joblib
from load_data import *
from time import monotonic
from matplotlib import pyplot as plt
from numpy.linalg import norm
import sys
from os import path

input_shape = [None, img_input_shape[0], img_input_shape[1], 3]
n_filters = [32, 64, 64, 128]
filter_sizes = [3, 3, 3, 3]
batch_size = 64
dev = 0.003
training_iters = 530000

conv_layers = [
    [ [1, 3, 3, 32] ],
    [ [2, 3, 3, 64] ],
    [ [3, 3, 3, 64] ], 
    [ [4, 3, 3, 128] ], 
]
fc1_size = 256

weight_decay = 0.004
base_lr = 0.001
stepsize = 5000
gamma = 0.8

data_dump = 'data.npz'


def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def convolitions(data_in, conv_layers):
    print("shape_in =", data_in.get_shape().as_list())
    d_in = data_in.get_shape().as_list()[3:]
    d_in = 1 if len(d_in) == 0 else d_in[0]
    for conv_block in conv_layers:
        for j, (i, r, c, d_out) in enumerate(conv_block):
            w = tf.get_variable("conv%d_%d_W" % (i, j + 1),
                                shape=[r, c, d_in, d_out],
                                initializer=tf.truncated_normal_initializer(stddev=dev),
                                trainable=True)
            b = tf.get_variable("conv%d_%d_b" % (i, j + 1),
                                shape=[d_out],
                                initializer=tf.constant_initializer(dev),
                                trainable=True)
            conv = tf.nn.conv2d(data_in, w, strides=[1, 1, 1, 1], padding='SAME')
            conv = tf.nn.relu(tf.nn.bias_add(conv, b))
            d_in = d_out
            data_in = conv
        data_in = max_pool(data_in, 2)    
        print("mp shape =", data_in.get_shape().as_list())
    return data_in


def build_graph():    
    x_ph = tf.placeholder(tf.uint8, input_shape, 'x')
    y_ph = tf.placeholder(tf.float32, [None, n_landmarks, 2], 'y')
    x_float = tf.cast(x_ph, tf.float32) / 255
    
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    conv = convolitions(x_float, conv_layers)
    
    
    # Fully connected layer
    data_size = np.prod(conv.get_shape().as_list()[1:])
    w = tf.Variable(tf.truncated_normal([data_size, fc1_size], stddev=dev),\
            name="fc1_W")
    b = tf.Variable(tf.constant(dev, shape=[fc1_size]),\
            name="fc1_b")
    dense1 = tf.reshape(conv, [-1, data_size])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, w), b))
    
    
    w = tf.Variable(tf.truncated_normal([dense1.get_shape().as_list()[1], 2 * n_landmarks], \
        stddev=dev), name="fc2_W")
    b = tf.Variable(tf.constant(dev, shape=[2 * n_landmarks]), name="fc2_b")
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, w), b))
    dense2 = tf.nn.dropout(dense2, keep_prob)
    
    prediction = tf.reshape(dense2, (-1, n_landmarks, 2))

    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    regularizers = weight_decay * l2_loss
    # l2 loss
    loss_x = tf.reduce_sum(tf.squared_difference(prediction, y_ph), [1, 2])
    cost = tf.reduce_mean(loss_x) + regularizers

    return {'cost': cost, 'x': x_ph, 'y': y_ph, 'pred': prediction,
            'keep_prob': keep_prob,
            'train': phase_train}


def train():
    try:
        data = np.load(data_dump)
        train_x, train_y = data['train_x'], data['train_y']
    except (KeyError, FileNotFoundError):
        data = None
    if data is None:
        train_x, train_y, train_names = load_data(sys.argv[1])
        np.savez(data_dump, train_x=train_x, train_y=train_y, 
                 train_names=train_names)
    print('train_x', train_x.shape, train_x.dtype)
    #for i in range(100):
    #    batch_show(train_x[i*64:i*64+64])
    # y normalization
    train_y[..., 0] /= img_input_shape[1]
    train_y[..., 1] /= img_input_shape[0]
    train_y = train_y * 2 - 1
    
    bg = BatchGenerator(train_x, train_y)
    
    with tf.device("/gpu:0"):
        conv_net = build_graph()
        batch = tf.Variable(0, dtype=tf.int32)
        learning_rate = tf.train.exponential_decay(base_lr, batch * batch_size, stepsize, gamma, staircase=True)
        optimizer = tf.train.AdamOptimizer(
            learning_rate).minimize(conv_net['cost'], global_step=batch)
    save_step = 10000
    saver = tf.train.Saver()
     
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=5)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()

        # Ensure no more changes to graph
        tf.get_default_graph().finalize()

        # Start up the queues for handling the image pipeline
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        batch_i = 0

        while batch_i * batch_size < training_iters:
            batch_xs, batch_label = bg.next_batch(batch_size)

            train_cost, pred = sess.run([conv_net['cost'], conv_net['pred'], optimizer], feed_dict={
                conv_net['x']: batch_xs, conv_net['y']: batch_label, conv_net['train']: True,
                conv_net['keep_prob']: 0.5})[:2]
            if batch_i % 40 == 0:
                print("Iter %d: loss=%.3f" % (batch_i * batch_size, train_cost))
                print("batch %d, lr %.5f" % tuple(sess.run([batch, learning_rate])))
                #for line in np.hstack((pred[3], batch_label[3], (pred[3] - batch_label[3]) ** 2)):
                #    print(", ".join(map(lambda x: "%.3f" % x, line))) 
            if batch_i % save_step == 0:
                # Save the variables to disk.
                saver.save(sess, "./models/" + 'conv_net.ckpt',
                           global_step=batch_i,
                           write_meta_graph=False)
            batch_i += 1
        
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train()




