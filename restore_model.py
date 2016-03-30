import tensorflow as tf
import numpy as np

v1 = tf.Variable(np.random.rand(100,20),name='v1')
v2 = tf.Variable(np.random.rand(20,5),name='v2')
prod = tf.matmul(v1,v2)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    print "Model restored."
    result = sess.run(prod)
    print result
