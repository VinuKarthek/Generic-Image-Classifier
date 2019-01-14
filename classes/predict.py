'''
Created on Sep 20, 2018

@author: Vinu Karthek
'''

import tensorflow as tf
import numpy as np
import tf_basics as tfb
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


class predict(object):
    '''
    #load saved models & predict images
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    
    def load_settings_from_json(self, json_file):
        return    
    
    def predict(self):
        with tf.name_scope('inputs'):
            X = tf.placeholder(tf.float32, [None,784], name ='X')
            W = tf.Variable(tf.zeros([784,10]), name = 'W')
            b = tf.Variable(tf.zeros([10]), name ='b')
        with tf.name_scope('layer1'):
            Y = tf.nn.softmax(tf.matmul(X, W) + b) 
        saver = tf.train.Saver()
        with tf.Session() as sess:
            #saver.restore(sess, 'my_net/Linear_Regression_net.ckpt')
            sess = tfb.load_session(saver, sess)
            batch = mnist.train.next_batch(1)
            result = sess.run(Y, feed_dict = {X:batch[0]})#,Y_:mnist.test.labels})
            print( "The Prediction is:"+ str(np.argmax(result)))
            #test_data = {X:mnist.test.images,Y_:mnist.test.labels}
            #a,c = sess.run([accuracy, cross_entrophy],feed_dict = test_data)
            #print(a,c)
            #plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
            
        
            plotData = batch[0]
            plotData = plotData.reshape(28, 28)
            plt.gray() # use this line if you don't want to see it in color
            plt.imshow(plotData)
            plt.show()