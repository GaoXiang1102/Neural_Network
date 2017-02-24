import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# The configuration of the Convolutional Neural Network is defined here for convenience,
#  so we can easily find and change these numbers and re-run the Notebook.
# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# The MNIST data-set is about 12 MB and will be downloaded automatically if it is not located in the given path.
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

# The MNIST data-set has now been loaded and consists of 70,000 images and associated labels (i.e.
# classifications of the images). The data-set is split into 3 mutually exclusive sub-sets. We will only use the
# training and test-sets.
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))




# The class-labels are One-Hot encoded, which means that each label is a vector with 10 elements,
# all of which are zero except for one element. The index of this one element is the class-number,
# that is, the digit shown in the associated image. We also need the class-numbers as integers for
# the test-set, so we calculate it now.
data.test.cls = np.argmax(data.test.labels, axis=1)

# The data dimensions are used in several places in the source-code below. They are defined once so
# we can use these variables instead of numbers throughout the source-code below.

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




# Functions for creating new TensorFlow variables in the given shape and initializing
# them with random values. Note that the initialization is not actually done at this point,
# it is merely being defined in the TensorFlow graph.
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))




# It is assumed that the input is a 4-dim tensor with the following dimensions:
# Image number.
# Y-axis of each image.
# X-axis of each image.
# Channels of each image.
# Note that the input channels may either be colour-channels, or it may be filter-channels if
# the input is produced from a previous convolutional layer.

# The output is another 4-dim tensor with the following dimensions:
# Image number, same as input.
# Y-axis of each image. If 2x2 pooling is used, then the height and width of the input images is divided by 2.
# X-axis of each image. Ditto.
# Channels produced by the convolutional filters.
def new_conv_layer(input,  # The previous layer.
                    num_input_channels,  # Num. channels in prev. layer.
                    filter_size,  # Width and height of each filter.
                    num_filters,  # Number of filters.
                    use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights





# A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected layers
# after the convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can be used as
# input to the fully-connected layer.
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features




# This function creates a new fully-connected layer in the computational graph for TensorFlow.
# Nothing is actually calculated here, we are just adding the mathematical formulas to the TensorFlow graph.
# It is assumed that the input is a 2-dim tensor of shape [num_images, num_inputs]. The output is a 2-dim tensor
#  of shape [num_images, num_outputs].
def new_fc_layer(input,  # The previous layer.
                num_inputs,  # Num. inputs from prev. layer.
                num_outputs,  # Num. outputs.
                use_relu=True):  # Use Rectified Linear Unit (ReLU)?


    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer





# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
# The convolutional layers expect x to be encoded as a 4-dim tensor so we have to reshape it so its shape
# is instead [num_images, img_height, img_width, num_channels]. Note that img_height == img_width == img_size
# and num_images can be inferred automatically by using -1 for the size of the first dimension. So the reshape operation is:
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
# Next we have the placeholder variable for the true labels associated with the images that were input in the placeholder
#  variable x. The shape of this placeholder variable is [None, num_classes] which means it may hold an arbitrary number
# of labels and each label is a vector of length num_classes which is 10 in this case.
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
# We could also have a placeholder variable for the class-number, but we will instead calculate it using argmax.
# Note that this is a TensorFlow operator so nothing is calculated at this point.
y_true_cls = tf.argmax(y_true, dimension=1)

# Convolutional Layer 1
# Create the first convolutional layer. It takes x_image as input and creates num_filters1 different filters, each having
# width and height equal to filter_size1. Finally we wish to down-sample the image so it is half the size by using 2x2 max-pooling.
layer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)



# Convolutional Layer 2
# Create the second convolutional layer, which takes as input the output from the first convolutional layer. The number of input
# channels corresponds to the number of filters in the first convolutional layer.
layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)



# Flatten Layer
# The convolutional layers output 4-dim tensors. We now wish to use these as input in a fully-connected network, which requires
# for the tensors to be reshaped or flattened to 2-dim tensors.
layer_flat, num_features = flatten_layer(layer_conv2)


# Fully-Connected Layer 1
# Add a fully-connected layer to the network. The input is the flattened layer from the previous convolution. The number of
# neurons or nodes in the fully-connected layer is fc_size. ReLU is used so we can learn non-linear relations.
layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)



# Fully-Connected Layer 2
# Add another fully-connected layer that outputs vectors of length 10 for determining which of the 10 classes the input
# image belongs to. Note that ReLU is not used in this layer.
layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)



# Predicted Class
# The second fully-connected layer estimates how likely it is that the input image belongs to each of the 10 classes.
# However, these estimates are a bit rough and difficult to interpret because the numbers may be very small or large,
# so we want to normalize them so that each element is limited between zero and one and the 10 elements sum to one.
# This is calculated using the so-called softmax function and the result is stored in y_pred.
y_pred = tf.nn.softmax(layer_fc2)
# The class-number is the index of the largest element.
y_pred_cls = tf.argmax(y_pred, dimension=1)




# Cost-function to be optimized
# TensorFlow has a built-in function for calculating the cross-entropy. Note that the function calculates the softmax
# internally so we must use the output of layer_fc2 directly rather than y_pred which has already had the softmax applied.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)



# In order to use the cross-entropy to guide the optimization of the model's variables we need a single scalar value,
# so we simply take the average of the cross-entropy for all the image classifications.
cost = tf.reduce_mean(cross_entropy)


# Optimization Method
# Now that we have a cost measure that must be minimized, we can then create an optimizer. In this case it is the
# AdamOptimizer which is an advanced form of Gradient Descent.
# Note that optimization is not performed at this point. In fact, nothing is calculated at all, we just add the
# optimizer-object to the TensorFlow graph for later execution.
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)



# Performance Measures
# This is a vector of booleans whether the predicted class equals the true class of each image.
correct_prediction = tf.equal(y_pred_cls, y_true_cls)


# This calculates the classification accuracy by first type-casting the vector of booleans to floats,
#  so that False becomes 0 and True becomes 1, and then calculating the average of these numbers.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Once the TensorFlow graph has been created, we have to create a TensorFlow session which is used to execute the graph.
session = tf.Session()
# The variables for weights and biases must be initialized before we start optimizing them.
session.run(tf.global_variables_initializer())



# There are 55,000 images in the training-set. It takes a long time to calculate the gradient of the model using all these images.
#  We therefore only use a small batch of images in each iteration of the optimizer.
# If your computer crashes or becomes very slow because you run out of RAM, then you may try and lower this number,
# but you may then need to perform more optimization iterations.
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
    plt.figure()
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
# Function for printing the classification accuracy on the test-set.
# It takes a while to compute the classification for all the images in the test-set,
# that's why the results are re-used by calling the above functions directly from this function,
# so the classifications don't have to be recalculated by each function.

# Note that this function can use a lot of computer memory, which is why the test-set is split into smaller batches.
#  If you have little RAM in your computer and it crashes, then you can try and lower the batch-size.
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
# The accuracy on the test-set is very low because the model variables have only been initialized and not optimized at all,
# so it just classifies the images randomly.
print_test_accuracy()


# Performance after 1 optimization iteration
# The classification accuracy does not improve much from just 1 optimization iteration, because the learning-rate
# for the optimizer is set very low.
optimize(num_iterations=1)
print_test_accuracy()



# Performance after 100 optimization iterations
optimize(num_iterations=99) # We already performed 1 iteration above.
print_test_accuracy()



# After 1000 optimization iterations, the model has greatly increased its accuracy on the test-set to more than 90%.
optimize(num_iterations=900) # We performed 100 iterations above.
print_test_accuracy()



# Visualization of Weights and Layers
# In trying to understand why the convolutional neural network can recognize handwritten digits, we will now
# visualize the weights of the convolutional filters and the resulting output images.
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




# Helper-function for plotting the output of a convolutional layer
def plot_conv_layer(layer, image):
    # Assume layer is a TensorFlow op that outputs a 4-dim tensor
    # which is the output of a convolutional layer,
    # e.g. layer_conv1 or layer_conv2.

    # Create a feed-dict containing just one image.
    # Note that we don't need to feed y_true because it is
    # not used in this calculation.
    feed_dict = {x: [image]}

    # Calculate and retrieve the output values of the layer
    # when inputting that image.
    values = session.run(layer, feed_dict=feed_dict)

    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_filters)))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



# Helper-function for plotting an image.
def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()



# Plot an image from the test-set which will be used as an example below.
image1 = data.test.images[0]
plot_image(image1)

# Now plot the filter-weights for the first convolutional layer.
# Note that positive weights are red and negative weights are blue.
plot_conv_weights(weights=weights_conv1)

# Applying each of these convolutional filters to the first input image gives
# the following output images, which are then used as input to the second convolutional layer.
# Note that these images are down-sampled to 14 x 14 pixels which is half the resolution of the original input image.
plot_conv_layer(layer=layer_conv1, image=image1)


# Convolution Layer 2
# Now plot the filter-weights for the second convolutional layer.
# There are 16 output channels from the first conv-layer, which means there are 16 input channels to the second conv-layer.
# The second conv-layer has a set of filter-weights for each of its input channels. We start by plotting the filter-weigths
# for the first channel. Note again that positive weights are red and negative weights are blue.
plot_conv_weights(weights=weights_conv2, input_channel=0)
# There are 16 input channels to the second convolutional layer, so we can make another 15 plots of filter-weights like this.
#  We just make one more with the filter-weights for the second channel.
plot_conv_weights(weights=weights_conv2, input_channel=1)
# plot the output of the second convolutional layer.
plot_conv_layer(layer=layer_conv2, image=image1)