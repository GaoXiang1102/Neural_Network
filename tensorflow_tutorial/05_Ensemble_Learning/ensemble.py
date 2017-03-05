import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

# We will train 5 neural networks on different training-sets that are selected at random.
# First we combine the original training- and validation-sets into one big set. This is done
#  for both the images and the labels.
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)

# Size of the combined data-set.
combined_size = len(combined_images)
# Define the size of the training-set used for each neural network. You can try and change this.
train_size = int(0.8 * combined_size)
# We do not use a validation-set during training, but this would be the size.
validation_size = combined_size - train_size

# Helper-function for splitting the combined data-set into a random training- and validation-set.
def random_training_set():
    # Create a randomized index into the full / combined training-set.
    idx = np.random.permutation(combined_size)

    # Split the random index into training- and validation-sets.
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    # Select the images and labels for the new training-set.
    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    # Select the images and labels for the new validation-set.
    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    # Return the new training- and validation-sets.
    return x_train, y_train, x_validation, y_validation


# The data dimensions are used in several places in the source-code below.
# They are defined once so we can use these variables instead of numbers throughout the source-code below.
# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10


# Function used to plot 9 images in a 3x3 grid, and writing the true and predicted classes below each image.
def plot_images(images,  # Images to plot, 2-d array.
                cls_true,  # True class-no for images.
                ensemble_cls_pred=None,  # Ensemble predicted class-no.
                best_cls_pred=None):  # Best-net predicted class-no.


    assert len(images) == len(cls_true)

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # For each of the sub-plots.
    for i, ax in enumerate(axes.flat):

        # There may not be enough images for all sub-plots.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i].reshape(img_shape), cmap='binary')

            # Show true and predicted classes.
            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net: {2}"
                xlabel = msg.format(cls_true[i],
                                    ensemble_cls_pred[i],
                                    best_cls_pred[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Tensorflow Graph
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


# Neural Network
x_pretty = pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)


optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)


y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
saver = tf.train.Saver(max_to_keep=100)
save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)


# TensorFlow Run
session = tf.Session()

def init_variables():
    session.run(tf.initialize_all_variables())

train_batch_size = 64


# Function for selecting a random training-batch of the given size.
def random_batch(x_train, y_train):
    # Total number of images in the training-set.
    num_images = len(x_train)

    # Create a random index into the training-set.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = x_train[idx, :]  # Images.
    y_batch = y_train[idx, :]  # Labels.

    # Return the batch.
    return x_batch, y_batch


# Function for performing a number of optimization iterations so as to gradually
# improve the variables of the network layers. In each iteration, a new batch of data
# is selected from the training-set and then TensorFlow executes the optimizer using those training samples.
# The progress is printed every 100 iterations.
def optimize(num_iterations, x_train, y_train):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch(x_train, y_train)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations and after last iteration.
        if i % 100 == 0:
            # Calculate the accuracy on the training-batch.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Status-message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Create ensemble of neural networks
num_networks = 5
num_iterations = 1000

if True:
    # For each of the neural networks.
    for i in range(num_networks):
        print("Neural network: {0}".format(i))

        # Create a random training-set. Ignore the validation-set.
        x_train, y_train, _, _ = random_training_set()

        # Initialize the variables of the TensorFlow graph.
        session.run(tf.global_variables_initializer())

        # Optimize the variables using this training-set.
        optimize(num_iterations=num_iterations,
                 x_train=x_train,
                 y_train=y_train)

        # Save the optimized variables to disk.
        saver.save(sess=session, save_path=get_save_path(i))

        # Print newline.
        print()

# This function calculates the predicted labels of images, that is, for each image it calculates a vector of length 10
# indicating which of the 10 classes the image is.


# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

def predict_labels(images):
    # Number of images.
    num_images = len(images)

    # Allocate an array for the predicted labels which
    # will be calculated in batches and filled into this array.
    pred_labels = np.zeros(shape=(num_images, num_classes),
                           dtype=np.float)

    # Now calculate the predicted labels for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images between index i and j.
        feed_dict = {x: images[i:j, :]}

        # Calculate the predicted labels using TensorFlow.
        pred_labels[i:j] = session.run(y_pred, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    return pred_labels


# Calculate a boolean array whether the predicted classes for the images are correct.
def correct_prediction(images, labels, cls_true):
    # Calculate the predicted labels.
    pred_labels = predict_labels(images=images)

    # Calculate the predicted class-number for each image.
    cls_pred = np.argmax(pred_labels, axis=1)

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct


# Calculate a boolean array whether the images in the test-set are classified correctly.
def test_correct():
    return correct_prediction(images = data.test.images,
                              labels = data.test.labels,
                              cls_true = data.test.cls)


# Calculate a boolean array whether the images in the validation-set are classified correctly.
def validation_correct():
    return correct_prediction(images = data.validation.images,
                              labels = data.validation.labels,
                              cls_true = data.validation.cls)


# This function calculates the classification accuracy given a boolean array whether each image
#  was correctly classified. E.g. classification_accuracy([True, True, False, False, False]) = 2/5 = 0.4
def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    return correct.mean()



# Calculate the classification accuracy on the test-set.
def test_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the test-set.
    correct = test_correct()

    # Calculate the classification accuracy and return it.
    return classification_accuracy(correct)



# Calculate the classification accuracy on the original validation-set.
def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    correct = validation_correct()

    # Calculate the classification accuracy and return it.
    return classification_accuracy(correct)


# Function for calculating the predicted labels for all the neural networks in the ensemble. The labels are combined further below.
def ensemble_predictions():
    # Empty list of predicted labels for each of the neural networks.
    pred_labels = []

    # Classification accuracy on the test-set for each network.
    test_accuracies = []

    # Classification accuracy on the validation-set for each network.
    val_accuracies = []

    # For each neural network in the ensemble.
    for i in range(num_networks):
        # Reload the variables into the TensorFlow graph.
        saver.restore(sess=session, save_path=get_save_path(i))

        # Calculate the classification accuracy on the test-set.
        test_acc = test_accuracy()

        # Append the classification accuracy to the list.
        test_accuracies.append(test_acc)

        # Calculate the classification accuracy on the validation-set.
        val_acc = validation_accuracy()

        # Append the classification accuracy to the list.
        val_accuracies.append(val_acc)

        # Print status message.
        msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
        print(msg.format(i, val_acc, test_acc))

        # Calculate the predicted labels for the images in the test-set.
        # This is already calculated in test_accuracy() above but
        # it is re-calculated here to keep the code a bit simpler.
        pred = predict_labels(images=data.test.images)

        # Append the predicted labels to the list.
        pred_labels.append(pred)

    return np.array(pred_labels), \
           np.array(test_accuracies), \
           np.array(val_accuracies)


pred_labels, test_accuracies, val_accuracies = ensemble_predictions()

# Summarize the classification accuracies on the test-set for the neural networks in the ensemble.
print("Mean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))


# There are different ways to calculate the predicted labels for the ensemble. One way is to calculate
# the predicted class-number for each neural network, and then select the class-number with most votes.
# But this requires a large number of neural networks relative to the number of classes.

# The method used here is instead to take the average of the predicted labels for all the networks in the ensemble.
# This is simple to calculate and does not require a large number of networks in the ensemble.
ensemble_pred_labels = np.mean(pred_labels, axis=0)


# The ensemble's predicted class number is then the index of the highest number in the label, which is calculated using argmax as usual.
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)


# Boolean array whether each of the images in the test-set was correctly classified by the ensemble of neural networks.
ensemble_correct = (ensemble_cls_pred == data.test.cls)

# Negate the boolean array so we can use it to lookup incorrectly classified images.
ensemble_incorrect = np.logical_not(ensemble_correct)

# Now we find the single neural network that performed best on the test-set.
# First list the classification accuracies on the test-set for all the neural networks in the ensemble.
print test_accuracies

# The index of the neural network with the highest classification accuracy.
best_net = np.argmax(test_accuracies)

print 'the best neural network on the test accuracy: ' + str(best_net)

# The best neural network's classification accuracy on the test-set.
print "the best neural network's classification accuracy on the test-set is: " + str(test_accuracies[best_net])

# Predicted labels of the best neural network.
best_net_pred_labels = pred_labels[best_net, :, :]

# The predicted class-number.
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)

# Boolean array whether the best neural network classified each image in the test-set correctly.
best_net_correct = (best_net_cls_pred == data.test.cls)

# Boolean array whether each image is incorrectly classified.
best_net_incorrect = np.logical_not(best_net_correct)


# Comparison of ensemble vs. the best single network
# The number of images in the test-set that were correctly classified by the ensemble.
print "The number of images in the test-set that were correctly classified by the ensemble:" + str(np.sum(ensemble_correct))

# The number of images in the test-set that were correctly classified by the best neural network.
print "The number of images in the test-set that were correctly classified by the best neural network:" + str(np.sum(best_net_correct))


# Boolean array whether each image in the test-set was correctly classified by the ensemble and incorrectly classified
# by the best neural network.
ensemble_better = np.logical_and(best_net_incorrect,
                                 ensemble_correct)

print "ensemble better:" + str(ensemble_better.sum())

# Boolean array whether each image in the test-set was correctly classified by the best single network and
# incorrectly classified by the ensemble.
best_net_better = np.logical_and(best_net_correct,
                                 ensemble_incorrect)
print "best network better:" + str(best_net_better.sum())


# Function for plotting images from the test-set and their true and predicted class-numbers.
def plot_images_comparison(idx):
    plot_images(images=data.test.images[idx, :],
                cls_true=data.test.cls[idx],
                ensemble_cls_pred=ensemble_cls_pred[idx],
                best_cls_pred=best_net_cls_pred[idx])


# Function for printing the predicted labels.
def print_labels(labels, idx, num=1):
    # Select the relevant labels based on idx.
    labels = labels[idx, :]

    # Select the first num labels.
    labels = labels[0:num, :]

    # Round numbers to 2 decimal points so they are easier to read.
    labels_rounded = np.round(labels, 2)

    # Print the rounded labels.
    print(labels_rounded)


# Function for printing the predicted labels for the ensemble of neural networks.
def print_labels_ensemble(idx, **kwargs):
    print_labels(labels=ensemble_pred_labels, idx=idx, **kwargs)


# Function for printing the predicted labels for the best single network.
def print_labels_best_net(idx, **kwargs):
    print_labels(labels=best_net_pred_labels, idx=idx, **kwargs)


# Function for printing the predicted labels of all the neural networks in the ensemble. This only prints the labels for the first image.
def print_labels_all_nets(idx):
    for i in range(num_networks):
        print_labels(labels=pred_labels[i, :, :], idx=idx, num=1)


# Plot examples of images that were correctly classified by the ensemble and incorrectly classified by the best single network.
plot_images_comparison(idx=ensemble_better)

# The ensemble's predicted labels for the first of these images (top left image):
print_labels_ensemble(idx=ensemble_better, num=1)


# The best network's predicted labels for the first of these images:
print_labels_best_net(idx=ensemble_better, num=1)

# The predicted labels of all the networks in the ensemble, for the first of these images:
print_labels_all_nets(idx=ensemble_better)


# Now plot examples of images that were incorrectly classified by the ensemble but correctly classified by the best single network.
plot_images_comparison(idx=best_net_better)

# The ensemble's predicted labels for the first of these images (top left image):
print_labels_ensemble(idx=best_net_better, num=1)

# The best single network's predicted labels for the first of these images:
print_labels_best_net(idx=best_net_better, num=1)


# The predicted labels of all the networks in the ensemble, for the first of these images:
print_labels_all_nets(idx=best_net_better)




