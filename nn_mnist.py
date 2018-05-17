import gzip
import _pickle

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = _pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set


# ---------------- Visualizing some element of the MNIST dataset --------------

#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#plt.imshow(train_x[58].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample


x_data = train_x
y_data = one_hot(train_y, 10)
x_data_v = valid_x
y_data_v = one_hot(valid_y, 10)
x_data_t = test_x
y_data_t = one_hot(test_y, 10)

x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 14)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(14)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(14, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print( "----------------------")
print( "   Start training...  ")
print( "----------------------")

batch_size = 20
error_prev = 1
error_act = 2
error_validacion = []
error_entrenamiento = []

while (abs((error_act-error_prev)/error_prev)>0.001):
    print("Entrenamiento")
    for jj in range(int(len(x_data) / batch_size)):
        batch_xs = x_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    error = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    error_entrenamiento.append(error)
    print("Error: ", error)
    print("Validación --")
    error_prev = error_act
    error_act = sess.run(loss, feed_dict={x: x_data_v, y_: y_data_v})
    error_validacion.append(error_act)
    print("Error: ", error_act)

    #print( "Epoch #:",(y epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))
    #result = sess.run, feed_dict={x: batch_xs})
    #for b, r in zip(batch_ys, result):
        #print( b, "-->", r)
    print( "----------------------------------------------------------------------------------")

print("Test --------")
contador = 0
result_test = sess.run(y, feed_dict={x: x_data_t})
for valor_r, valor_e in zip(y_data_t, result_test):
    if (np.argmax(valor_r)==np.argmax(valor_e)):
        contador += 1
print("Porcentaje de acierto: ", contador/len(test_y)*100, "%")

plt.plot(error_validacion)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error de validación")

plt.show()