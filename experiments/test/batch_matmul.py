import tensorflow as tf



A = tf.Variable(tf.ones(shape=(10, 2, 3)))
B = tf.constant([[1.0,2,2],[3,4,4]])

coords_uv = tf.constant([3,4])
coords_uv_ = tf.concat([tf.expand_dims(coords_uv[0], 0), tf.expand_dims(coords_uv[1], 0)], 0)
#C = tf.matmul(A, B_exp)
C = A/B
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(coords_uv))
    print(sess.run(coords_uv_))