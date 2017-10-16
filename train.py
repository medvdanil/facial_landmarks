import tensorflow as tf
import numpy as np
from load_data import *
from time import monotonic
from matplotlib import pyplot as plt
from numpy.linalg import norm
import sys
from os import path

input_shape = [None, img_input_shape[0], img_input_shape[1], 3]
batch_size = 64
dev = 0.05
training_iters = 600000
dropout = 0.5

conv_layers = [
    [3, 3, 32, 3],
    [3, 3, 64, 3],
    [3, 3, 64, 2], 
    [2, 2, 128, 0], 
]
fc_sizes = [256, 512]

weight_decay = 0.0001
base_lr = 0.001
stepsize = 30000
gamma = 0.8
momentum = 0.9

data_dump = 'data.npz' #48_nojit.npz'
model_file = './models/conv_net.ckpt'

def max_pool(img, k, stride):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='VALID')


def convolitions(data_in, conv_layers):
    print("shape_in =", data_in.get_shape().as_list())
    d_in = data_in.get_shape().as_list()[3:]
    d_in = 1 if len(d_in) == 0 else d_in[0]
    
    for i, (r, c, d_out, mp_k) in enumerate(conv_layers):
        w = tf.get_variable("conv%d_W" % i, shape=[r, c, d_in, d_out],
                            initializer=tf.truncated_normal_initializer(stddev=dev),
                            trainable=True)
        b = tf.get_variable("conv%d_b" % i, shape=[d_out],
                            initializer=tf.constant_initializer(dev),
                            trainable=True)
        conv = tf.nn.conv2d(data_in, w, strides=[1, 1, 1, 1], padding='VALID')
        conv = tf.nn.relu(tf.nn.bias_add(conv, b))
        if i == 2:
            conv2 = conv
        d_in = d_out
        print("conv shape =", conv.get_shape().as_list(), conv.dtype)
        if mp_k > 0:
            data_in = max_pool(conv, mp_k, 2)
            print("mp shape =", data_in.get_shape().as_list())
        else:
            data_in = conv
        
    #return tf.concat([tf.reshape(data_in, [-1, np.prod(data_in.get_shape().as_list()[1:])]), 
    #                  tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])], 1), conv2
    return data_in, conv2


def build_graph(n_landmarks=n_landmarks):    
    x_ph = tf.placeholder(tf.float32, input_shape, 'x')
    y_ph = tf.placeholder(tf.float32, [None, n_landmarks, 2], 'y')
    conv2 = x_ph
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    conv, conv2 = convolitions(x_ph, conv_layers)
    
    dense = conv
    data_size = np.prod(dense.get_shape().as_list()[1:])
    dense = tf.reshape(dense, [-1, data_size])
    # Fully connected layers
    for i, fc_size in enumerate(fc_sizes):
        data_size = np.prod(dense.get_shape().as_list()[1:])
        w = tf.Variable(tf.truncated_normal([data_size, fc_size], stddev=dev),\
                name="fc%d_W" % i)
        b = tf.Variable(tf.constant(dev, shape=[fc_size]),\
                name="fc%d_b" % i)
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w), b))

        dense = tf.concat([tf.reshape(dense, [-1, np.prod(dense.get_shape().as_list()[1:])]), 
                           tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])], 1)
    
    dense = tf.nn.dropout(dense, keep_prob)
    
    #dense = tf.concat([tf.reshape(dense, [-1, np.prod(dense.get_shape().as_list()[1:])]), 
    #                  tf.reshape(conv2, [-1, np.prod(conv2.get_shape().as_list()[1:])])], 1)
    w = tf.Variable(tf.truncated_normal([dense.get_shape().as_list()[1], 2 * n_landmarks], \
        stddev=dev), name="fc_final_W")
    b = tf.Variable(tf.constant(dev, shape=[2 * n_landmarks]), name="fc_final_b")
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense, w), b))
    #dense2 = tf.nn.dropout(dense2, keep_prob)
    
    prediction = tf.reshape(dense2, (-1, n_landmarks, 2))

    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    # l2 loss
    cost = tf.reduce_mean(tf.squared_difference(prediction, y_ph)) +  weight_decay * l2_loss

    return {'cost': cost, 'x': x_ph, 'y': y_ph, 'pred': prediction,
            'keep_prob': keep_prob, 'conv2':conv2}

def calc_score(sess, conv_net, test_x, test_y, test_wh, interocular=False):
    err_arr = np.zeros(0)
    loss_sum = 0.0
    for i in range((len(test_x) + batch_size - 1) // batch_size):
        loss, pred, conv2 = sess.run([conv_net['cost'], conv_net['pred'], conv_net['conv2']], feed_dict={
            conv_net['x']: test_x[i * batch_size:(i + 1) * batch_size], 
            conv_net['y']: test_y[i * batch_size:(i + 1) * batch_size],
            conv_net['keep_prob']: 1.0})
        if interocular:
            print(test_y.shape)
            err = np.sum((pred - test_y[i * batch_size:(i + 1) * batch_size]) ** 2, axis=2)
            idist = (test_y[i * batch_size:(i + 1) * batch_size, 44] - \
                     test_y[i * batch_size:(i + 1) * batch_size, 37]) ** 2
            print(idist.shape)
            err /= np.sum(idist, axis=1)[:, None]
            err = np.mean(np.sqrt(err), axis=1)
        else:
            err = ((pred - test_y[i * batch_size:(i + 1) * batch_size]) * \
                test_wh[i * batch_size:(i + 1) * batch_size, None,:]) ** 2
            err = np.mean(np.sqrt(np.sum(err, axis=2)), axis=1)
            err /= np.sqrt(np.prod(test_wh[i * batch_size:(i + 1) * batch_size], axis=1))
        #landmarks_batch_show(test_x[i * batch_size:(i + 1) * batch_size], pred * img_input_shape[0])
        err_arr = np.concatenate((err_arr, err))
        loss_sum += loss * len(pred)
    return loss_sum / len(test_x), err_arr


def train():
    try:
        data = np.load(data_dump)
    except (KeyError, FileNotFoundError):
        data = None
    if data is None:
        train_data = load_data(sys.argv[1], is_train=True)
        test_data = load_data(sys.argv[1], is_train=False)
        data = {**train_data, **test_data}
        np.savez(data_dump, **data)
    train_x, train_y = data['train_x'], data['train_y']
    test_x, test_y = data['test_x'], data['test_y']
    train_wh = data['train_wh']
    test_wh = data['test_wh']
    dlib_err = data['dlib_err']
    #draw_CED(dlib_err)
    
    print('train_x', train_x.shape, train_x.dtype)
    print(len(train_wh), len(train_y))
    #for i in range(100):
    #    landmarks_batch_show(train_x[i*batch_size:i*batch_size+batch_size], train_y[i*batch_size:i*batch_size+batch_size])
    #    batch_show(train_x[i*batch_size:i*batch_size+batch_size])
     
    # y normalization
    train_y[..., 0] /= img_input_shape[1]
    train_y[..., 1] /= img_input_shape[0]
    test_y[..., 0] /= img_input_shape[1]
    test_y[..., 1] /= img_input_shape[0]
    #train_y = train_y[:,[41, 46, 30, 57, 19, 24],:]
    #test_y = test_y[:,[41, 46, 30, 57, 19, 24],:]
    
    bg = BatchGenerator(train_x, train_y)
    
    #with tf.device("/gpu:0"):
    conv_net = build_graph()
    batch = tf.Variable(0, dtype=tf.int32)
    learning_rate = tf.train.exponential_decay(base_lr, batch * batch_size, stepsize, gamma, staircase=True)
    optimizer = tf.train.AdamOptimizer(
        learning_rate).minimize(conv_net['cost'], global_step=batch)
    #optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(conv_net['cost'], global_step=batch)
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
        
        loss_plot = []
        batch_i = 0
        try:
            while batch_i * batch_size < training_iters:
                batch_xs, batch_label = bg.next_batch(batch_size)

                sess.run(optimizer, feed_dict={conv_net['x']: batch_xs, conv_net['y']: batch_label, 
                                            conv_net['keep_prob']: dropout})
                if batch_i % 40 == 0:
                    train_loss, pred, conv2 = sess.run([conv_net['cost'], conv_net['pred'], conv_net['conv2']], feed_dict={
                        conv_net['x']: train_x[:batch_size], conv_net['y']: train_y[:batch_size],
                        conv_net['keep_prob']: 1.0})
                    
                    print("Iter %d: loss=%.6f" % (batch_i * batch_size, train_loss))
                    print("batch %d, lr %.5f" % tuple(sess.run([batch, learning_rate])))
                    
                if (batch_i + 1) % 200 == 0:
                    loss, err_arr = calc_score(sess, conv_net, test_x, test_y, test_wh)
                    #if i == 0:
                    #    for line in np.hstack((pred[3], test_y[3], (pred[3] - test_y[3]) ** 2)):
                    #        print(", ".join(map(lambda x: "%.3f" % x, line)))
                    #w1 = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv%d_b" % 1)[0])
                    #print(w1)
                    err_arr.dump('err_arr.dump')
                    loss_plot.append([batch_i * batch_size, loss])
                    if (batch_i + 1) % 400 == 0:
                        conv2 = np.transpose(conv2, [0, 3, 1, 2])
                        #conv2 = np.reshape(conv2, (-1,) + conv2.shape[2:])
                        toshow = conv2[:64]
                        toshow = np.stack([conv2, conv2, conv2], axis=-1)
                        print(toshow.min(), toshow.max())
                        toshow/=toshow.max()
                        for i in range(5):
                            print(np.mean((toshow[i] - toshow[0]) ** 2), np.mean((pred[i] - pred[0]) ** 2), 
                                np.mean((test_y[i] - test_y[0]) ** 2))
                            #batch_show(toshow[i])
                            
                    #if (batch_i + 1) % 2000 == 0:
                    #    draw_CED(err_arr, dlib_err)
                            
                    print("test loss = %.6f, norm err = %.6f" % (loss, err_arr.mean()))
                    print("model_saved to " + \
                        saver.save(sess, model_file))
                batch_i += 1
        except KeyboardInterrupt:
            pass
        loss_plot = np.array(loss_plot)
        plt.plot(loss_plot[:,0], loss_plot[:,1])
        plt.show()
        
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train()



#train 0.07547 64x64
#test 0.071184
#test 0.072645 48


