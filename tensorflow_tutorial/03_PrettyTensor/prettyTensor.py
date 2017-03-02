import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# We also need PrettyTensor.
import prettytensor as pt


# The MNIST data-set is about 12 MB and will be downloaded automatically if it is not located in the given path.
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

# The MNIST data-set has now been loaded and consists of 70,000 images and associated labels (i.e. classifications
#  of the images). The data-set is split into 3 mutually exclusive sub-sets. We will only use the training and test-sets in this tutorial.
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

# The class-labels are One-Hot encoded, which means that each label is a vector with 10 elements, all of which are zero except for one element.
data.test.cls = np.argmax(data.test.labels, axis=1)

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
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Get the first images from the test-set.
images = data.test.images[0:9]

# Get the true classes for those images.
cls_true = data.test.cls[0:9]

# Plot the images and labels using our helper-function above.
# plot_images(images=images, cls_true=cls_true)


# First we define the placeholder variable for the input images.
# This allows us to change the images that are input to the TensorFlow graph.
# This is a so-called tensor, which just means that it is a multi-dimensional array.
# he data-type is set to float32 and the shape is set to [None, img_size_flat],
# where None means that the tensor may hold an arbitrary number of images with each image being a vector of length img_size_flat.
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

# The convolutional layers expect x to be encoded as a 4-dim tensor so we have to reshape it so its shape is instead [num_images,
# img_height, img_width, num_channels]. Note that img_height == img_width == img_size and num_images can be inferred automatically
#  by using -1 for the size of the first dimension. So the reshape operation is:
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# Next we have the placeholder variable for the true labels associated with the images that were input in the placeholder variable x.
# The shape of this placeholder variable is [None, num_classes] which means it may hold an arbitrary number of labels and each label
# is a vector of length num_classes which is 10 in this case.
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')

# We could also have a placeholder variable for the class-number, but we will instead calculate it using argmax.
# Note that this is a TensorFlow operator so nothing is calculated at this point.
y_true_cls = tf.argmax(y_true, dimension=1)



# Graph Construction
# PrettyTensor Implementation
x_pretty = pt.wrap(x_image)
# Now that we have wrapped the input image in a PrettyTensor object,
# we can add the convolutional and fully-connected layers in just a few lines of source-code.
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)

# That's it! We have now created the exact same Convolutional Neural Network in a few simple lines of code that required many complex
# lines of code in the direct TensorFlow implementation. Using PrettyTensor instead of TensorFlow, we can clearly see the network
# structure and how the data flows through the network. This allows us to focus on the main ideas of the Neural Network rather
# than low-level implementation details.


# Getting the Weights
def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'weights' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.

    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


# Using this helper-function we can retrieve the variables. These are TensorFlow objects. In order to get the contents of the variables,
    # you must do something like: contents = session.run(weights_conv1) as demonstrated further below.
weights_conv1 = get_weights_variable(layer_name='layer_conv1')
weights_conv2 = get_weights_variable(layer_name='layer_conv2')


# PrettyTensor gave us the predicted class-label (y_pred) as well as a loss-measure that must be minimized, so as to improve the ability
# of the Neural Network to classify the input images.
# It is unclear from the documentation for PrettyTensor whether the loss-measure is cross-entropy or something else. But we now use the
#  AdamOptimizer to minimize the loss.
# Note that optimization is not performed at this point. In fact, nothing is calculated at all, we just add the optimizer-object to the
# TensorFlow graph for later execution.
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)


# Performance Measures
# We need a few more performance measures to display the progress to the user.
# First we calculate the predicted class number from the output of the Neural Network y_pred, which is a vector with 10 elements.
# The class number is the index of the largest element.
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Then we create a vector of booleans telling us whether the predicted class equals the true class of each image.
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

# The classification accuracy is calculated by first type-casting the vector of booleans to floats, so that False
# becomes 0 and True becomes 1, and then taking the average of these numbers.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Once the TensorFlow graph has been created, we have to create a TensorFlow session which is used to execute the graph.
session = tf.Session()

# The variables for weights and biases must be initialized before we start optimizing them.
session.run(tf.global_variables_initializer())


# There are 55,000 images in the training-set. It takes a long time to calculate the gradient of the model using all
# these images. We therefore only use a small batch of images in each iteration of the optimizer.
train_batch_size = 64

# Function for performing a number of optimization iterations so as to gradually improve the variables of the network layers.
# In each iteration, a new batch of data is selected from the training-set and then TensorFlow executes the optimizer using
# those training samples. The progress is printed every 100 iterations.

# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


# Function for plotting examples of images from the test-set that have been mis-classified.
def plot_example_errors(cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


# Helper-function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Helper-function for showing the performance
# Split the test-set into smaller batches of this size.
test_batch_size = 256

def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = data.test.images[i:j, :]

        # Get the associated labels.
        labels = data.test.labels[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = data.test.cls

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# Performance before any optimization
print_test_accuracy()


# Performance after 1 optimization iteration
optimize(num_iterations=1)
print_test_accuracy()

# Performance after 100 optimization iterations
optimize(num_iterations=99) # We already performed 1 iteration above.
print_test_accuracy(show_example_errors=True)


# Performance after 1000 optimization iterations
optimize(num_iterations=900) # We performed 100 iterations above.
print_test_accuracy(show_example_errors=True)

# Visualization of Weights and Layers
# When the Convolutional Neural Network was implemented directly in TensorFlow,
#  we could easily plot both the convolutional weights and the images that were output
#  from the different layers. When using PrettyTensor instead, we can also retrieve the weights as shown above,
#  but we cannot so easily retrieve the images that are output from the convolutional layers. So in the following we only plot the weights.

# Helper-function for plotting convolutional weights
def plot_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    w = session.run(weights)

    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(w)
    w_max = np.max(w)

    # Number of filters used in the conv. layer.
    num_filters = w.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = w[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Now plot the filter-weights for the first convolutional layer.
# Note that positive weights are red and negative weights are blue.
plot_conv_weights(weights=weights_conv1)

# Now plot the filter-weights for the second convolutional layer.
# There are 16 output channels from the first conv-layer,
# which means there are 16 input channels to the second conv-layer.
# The second conv-layer has a set of filter-weights for each of its input channels.
#  We start by plotting the filter-weigths for the first channel.

# Note again that positive weights are red and negative weights are blue.
plot_conv_weights(weights=weights_conv2, input_channel=0)

