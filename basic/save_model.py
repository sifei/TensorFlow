import tensorflow as tf
imprt numpy as np

v1 = tf.Variable(np.random.rand(100,20),name='v1')
v2 = tf.Variable(np.random.rand(20,5),name='v2')
prod = tf.matmul(v1,v2)

init_op = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    result = sess.run(prod)
    print result
    save_path = saver.save(sess, "model.ckpt")
    print "Model saved in file: ", save_path
