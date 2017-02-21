import tensorflow as tf
import numpy as np
from data_provider.data_providers import CIFAR100DataProvider
import matplotlib.pyplot as plt


train_data = CIFAR100DataProvider('train', batch_size=50)
valid_data = CIFAR100DataProvider('valid', batch_size=50)



def fully_connected_layer(inputs, input_dim, output_dim, nonlinearity=tf.nn.relu):
    weights = tf.Variable(
        tf.truncated_normal(
            [input_dim, output_dim], stddev=2. / (input_dim + output_dim)**0.5),
        'weights')
    biases = tf.Variable(tf.zeros([output_dim]), 'biases')
    outputs = nonlinearity(tf.matmul(inputs, weights) + biases)
    return outputs, weights



inputs = tf.placeholder(tf.float32, [None, train_data.inputs.shape[1]], 'inputs')
targets = tf.placeholder(tf.float32, [None, train_data.num_classes], 'targets')



num_hidden = 300

with tf.name_scope('fc-layer-1'):
    hidden_1, weights1 = fully_connected_layer(inputs, train_data.inputs.shape[1], num_hidden)
with tf.name_scope('fc-layer-2'):
    hidden_2, weights2 = fully_connected_layer(hidden_1, num_hidden, num_hidden)
with tf.name_scope('fc-layer-3'):
    hidden_3, weights3 = fully_connected_layer(hidden_2, num_hidden, num_hidden)
with tf.name_scope('output-layer'):
    outputs, weights4 = fully_connected_layer(hidden_3, num_hidden, train_data.num_classes, tf.identity)



beta = 2e-3
with tf.name_scope('error'):
    error = (tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(outputs, targets)) +
             beta * tf.nn.l2_loss(weights1) +
             beta * tf.nn.l2_loss(weights2) +
             beta * tf.nn.l2_loss(weights3) +
             beta * tf.nn.l2_loss(weights4))
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
        tf.float32))



with tf.name_scope('train'):
    train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(error)



init = tf.global_variables_initializer()



epoch_num = 100
train_epochs = np.linspace(1, epoch_num, num=epoch_num)
valid_epochs = []
train_error_set = []
train_accuracy_set = []
valid_error_set = []
valid_accuracy_set = []



with tf.Session() as sess:
    sess.run(init)
    for e in range(epoch_num):
        running_error = 0.
        running_accuracy = 0.
        for input_batch, target_batch in train_data:
            _, batch_error, batch_acc = sess.run(
                [train_step, error, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch})
            running_error += batch_error
            running_accuracy += batch_acc
        running_error /= train_data.num_batches
        running_accuracy /= train_data.num_batches
        train_error_set.insert(e, running_error)
        train_accuracy_set.insert(e, running_accuracy)
        print('End of epoch {0:02d}: err(train)={1:.3f} acc(train)={2:.3f}'
              .format(e + 1, running_error, running_accuracy))
        if (e + 1) % 5 == 0:
            valid_error = 0.
            valid_accuracy = 0.
            for input_batch, target_batch in valid_data:
                batch_error, batch_acc = sess.run(
                    [error, accuracy],
                    feed_dict={inputs: input_batch, targets: target_batch})
                valid_error += batch_error
                valid_accuracy += batch_acc
            valid_error /= valid_data.num_batches
            valid_accuracy /= valid_data.num_batches
            valid_error_set.append(valid_error)
            valid_accuracy_set.append(valid_accuracy)
            valid_epochs.append(e + 1)
            print('                 err(valid)={0:.3f} acc(valid)={1:.3f}'
                   .format(valid_error, valid_accuracy))


plt.figure(1)
plt.subplot(211)
plt.plot(train_epochs, train_error_set, label="$train error$",color="red",linewidth=1)
plt.plot(valid_epochs, valid_error_set, "bo", label="$valid error$")
plt.xlabel("epoch")
plt.ylabel("error")

plt.subplot(212)
plt.plot(train_epochs, train_accuracy_set, label="$train accuracy$",color="red",linewidth=1)
plt.plot(valid_epochs, valid_accuracy_set, "bo", label="$valid accuracy$")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.show()