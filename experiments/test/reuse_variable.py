import tensorflow as tf
def list(v):
    a = []
    a.append(v)
    v = tf.get_variable("v2", [1])
    a.append(v)
    a.append(v)
    return a
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    vl = list(v)
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    v1l = list(v1)
a = v+v1
b = a
c = v+v1
#assert v1 == v
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(v))
    print(sess.run(v1))