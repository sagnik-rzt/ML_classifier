import tensorflow as tf

a = tf.Variable(initial_value = [5,6,7,8,9,10], name="var1")
b = tf.constant(6)
c = tf.placeholder(dtype = tf.int32)
d = a.assign(a + tf.constant([1,1,1,1,1,1]))
counter = tf.Variable(initial_value = 0)

x = a * b + c
tf.add_to_collection(name="var1", value=x)


saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init)
    print(sess)
    result1 = sess.run(x, feed_dict = {c : [0,1,2,3,4,5]})
    print(result1)
    print("Counter is", sess.run(counter.assign_add(1)))
    print('\n')

    result2 = sess.run(d)
    print(result2)
    result3 = sess.run(d)
    print(result3)
    writer = tf.summary.FileWriter("output", sess.graph)
    print("Counter is", sess.run(counter.assign_add(1)))

    saver.save(sess, save_path="saver/test.model")
