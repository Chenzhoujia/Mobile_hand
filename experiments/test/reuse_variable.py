import tensorflow as tf
with tf.variable_scope('foo'):
    a1 = tf.get_variable('bar',[5])
    a2 = tf.get_variable('bar2', [5])
    a3 = tf.stack([a1,a2], axis=1, name="final_rxyz")
    #打印foo/bar:0
    print(a1.name)
    #output_node = tf.Variable(initial_value=a, name='final_rxyz_Variable')
    output_node = tf.add(a1, 0, name='final_rxyz_Variable')
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a3))
    print(sess.run(output_node))