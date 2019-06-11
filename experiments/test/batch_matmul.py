import tensorflow as tf



A = tf.Variable(tf.ones(shape=(10, 2, 3)))
B = tf.Variable(tf.random_normal(shape=(10, 1, 1))*2)
#C = tf.matmul(A, B_exp)
C = B*A
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(A))
    print(sess.run(B))
    print(sess.run(C))