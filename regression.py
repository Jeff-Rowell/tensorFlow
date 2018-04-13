import tensorflow as tf
import numpy as np
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt

g = tf.Graph()

with g.as_default():
    tf.set_random_seed(123)
    
    ## placeholders
    tf_x = tf.placeholder(shape=(None), dtype=tf.float32, name='tf_x')
    tf_y = tf.placeholder(shape=(None), dtype=tf.float32, name='tf_y')
    
    ## define the variable model parameters
    weight = tf.Variable(tf.random_normal(shape=(1,1), stddev=0.25), name='weight')
    bias = tf.Variable(0.0, name='bias')

    ## build the model
    y_hat = tf.add(weight * tf_x, bias, name='y_hat')

    ## compute the cost
    cost = tf.reduce_mean(tf.square(tf_y - y_hat), name='cost')

    ## train the model
    optim = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optim.minimize(cost, name='train_op')

## create a random toy dataset for regression
np.random.seed(0)

def make_random_data():
    x = np.random.uniform(low=-2, high=4, size=200)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, scale=(0.5 + t*t/3), size=None)
        y.append(r)
    return x, 1.726*x - 0.84 + np.array(y)

x,y = make_random_data()
plt.plot(x, y, 'o')
plt.show()

## train/test splits
x_train, y_train = x[:100], y[:100]
x_test, y_test = x[100:], y[100:]

with g.as_default():
    saver = tf.train.Saver()
    
n_epochs = 500
training_costs = []
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())

    ## train the model for n_epochs
    for e in range(n_epochs):
        c, _ = sess.run([cost, train_op], feed_dict={tf_x: x_train,
                                                     tf_y: y_train})
        training_costs.append(c)
        if not e % 50:
            print('Epoch %4d: %.4f' %(e,c))
    saver.save(sess, './trained-model')
            
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.plot(training_costs)
plt.show()

g2 = tf.Graph()
with tf.Session(graph=g2) as sess:
    new_saver = tf.train.import_meta_graph('./trained-model.meta')
    new_saver.restore(sess, './trained-model')
    y_pred = sess.run('y_hat:0',feed_dict={'tf_x:0': x_test})

x_arr = np.arange(-2, 4, 0.1)
g2 = tf.Graph()
with tf.Session(graph=g2) as sess:
    new_saver = tf.train.import_meta_graph('./trained-model.meta')
    new_saver.restore(sess, './trained-model')
    y_arr = sess.run('y_hat:0', feed_dict={'tf_x:0' : x_arr})

plt.figure()
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_train, y_train, 'bo')
plt.plot(x_test, y_test, 'bo', color='green', alpha=0.3)
plt.plot(x_arr, y_arr.T[:, 0], '-r', lw=3)
plt.show()
