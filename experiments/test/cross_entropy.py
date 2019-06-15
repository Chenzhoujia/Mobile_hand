import tensorflow as tf

# 熵就是系统的混乱度，举个例子就是弄清一个系统所需问问题的数量。p*log2(1/p)
# 交叉熵就是在给定真实分布下，用估计的分布消除系统不确定性的成本。
y_ = tf.constant([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]) # 3 * 2 这里的3是batchsize×像素数 2是一个像素的分类
y = tf.constant([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
ysoft = tf.nn.softmax(y)
cross_entropy = -tf.reduce_sum(y_ * tf.log(ysoft))

# do cross_entropy just one step
cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))

with tf.Session() as sess:
    print("step1:softmax result=")
    print(sess.run(ysoft))
    print("step2:cross_entropy result=")
    print(sess.run(cross_entropy))
    print("Function(softmax_cross_entropy_with_logits) result=")
    print(sess.run(cross_entropy2))
    print("cross_entropy_loss result=")
    print(sess.run(cross_entropy_loss))