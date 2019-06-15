import tensorflow as tf

s = tf.constant([0,0,0,1,1,1,1], dtype=tf.float32)

sm = tf.nn.softmax(s)

with tf.Session()as sess:

    print(sess.run(sm))

    print(sess.run(tf.argmax(sm)))