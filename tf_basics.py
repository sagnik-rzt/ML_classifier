import tensorflow as tf
import numpy as np

a = tf.Variable(initial_value = [5,6,7,8,9,10], name="var1")
b = tf.constant(6)
c = tf.placeholder(dtype = tf.int32)
d = a.assign(a + tf.constant([1,1,1,1,1,1]))
counter = tf.Variable(initial_value = 0)

x = tf.add(tf.multiply(a, b), c)
tf.add_to_collection(name="var1", value=x)


saver = tf.train.Saver()


init = tf.global_variables_initializer()

def main1():

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


a1 = tf.Variable(initial_value = [[1,2,3], [4,5,6]], dtype = tf.float32 )
b1 = tf.ones(shape = [3, 4])
c1 = tf.matmul(a1, b1)
d1 = tf.random_normal(shape = [10,20])
e1, e2, e3 = tf.split(d1, [9, 9, 2], axis = 1)
f1 = np.array([[1,2,3], [4,5,6]])
g1 = tf.convert_to_tensor(f1, dtype = tf.float32)
h1 = tf.matmul(g1, b1)

def main2():

    with tf.Session() as sess:
        saver.restore(sess, save_path = "saver/test.model")
        sess.run(tf.initialize_all_variables())
        print(sess.run(h1))



a2 = [[1,2,3], [4,5,6]]
b2 = [[7,8,9], [10,11,12]]
c2 = tf.concat([a2, b2], axis = 0)
d2 = tf.concat([a2, b2], axis = 1)
e2 = tf.reshape(a2, shape = [-1])
f2 = tf.reshape(b2, shape = [6,1])
g2 = tf.transpose(b2)
h2 = tf.reduce_mean(a2, axis = 0)
i2 = tf.reduce_mean(a2, axis = 1)
j2 = tf.argmax(a2, axis = 0)
k2 = tf.argmax(a2, axis = 1)
l2 = tf.argmin(a2, axis = 0)
m2 = tf.argmin(a2, axis = 1)

def main3():

    with tf.Session() as sess:
        print(sess.run(l2))
        print(sess.run(m2))

main3()