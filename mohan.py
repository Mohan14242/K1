import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Print information about the dataset
print("Training images shape:", train_images.shape)  # (60000, 28, 28)
print("Training labels shape:", train_labels.shape)  # (60000,)
print("Test images shape:", test_images.shape)        # (10000, 28, 28)
print("Test labels shape:", test_labels.shape)        # (10000,)

# Print sample images and their corresponding labels
print("Sample training image:")
print(train_images[0])  # Print the first training image (28x28 array of pixel values)
print("Label:", train_labels[0])  # Print the label corresponding to the first training image

# Similarly, you can print sample test images and labels
