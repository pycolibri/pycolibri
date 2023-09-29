import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio

class LossAndPlotCallback(tf.keras.callbacks.Callback):

    def __init__(self, dataset, num_samples=1, plot_graph=False):
        """
        Callback to display loss and predictions during training.

        Parameters
        ----------
        dataset : tf.data.Dataset
            The training dataset.
        num_samples : int, optional
            The number of random samples to display, default is 1.
        plot_graph : bool, optional
            Indicates whether to plot Ground Truth images and predictions, default is False.

        """

        super(LossAndPlotCallback, self).__init__()
        self.dataset = dataset
        self.num_samples = num_samples
        self.plot_graph = plot_graph

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to display loss and predictions.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        logs : dict, optional
            A dictionary containing training metrics, default is None.

        """

        # Calculate the average loss for this epoch
        epoch_loss = logs["loss"]
        epoch_num = epoch + 1

        # Print the average loss for this epoch
        print('\n\n' + f'Epoch {epoch_num} - Average Loss: {epoch_loss:.4f}' + '\n\n')

        # Calculate the encoder output for sampled images
        for _ in range(self.num_samples):
            index = tf.random.uniform(
                (), maxval=len(self.dataset), dtype=tf.int32
            )
            sampled_images = self.dataset[index]  # A list of images

            # Process each image individually
            for sampled_image in sampled_images:
                print('sampled_image.shape:', sampled_image.shape)
                sampled_image = tf.expand_dims(sampled_image, 0)
                prediction = self.model(sampled_image)  # Change to the correct layer index

                # Plot the input image vs. the network output
                if self.plot_graph:
                    num_bands = sampled_image.shape[3]  # Number of bands in the input image

                    # Create a figure with subplots for each input and output band
                    plt.figure(figsize=(num_bands, 2))  # Increase the figure size

                    # Subplots for input
                    for band in range(num_bands):
                        plt.subplot(2, num_bands, band + 1)
                        plt.imshow(sampled_image[0, :, :, band].numpy(), cmap='gray')
                        plt.title(f'In: {band + 1}')
                        plt.axis('off')  # Remove axes

                    # Subplots for output
                    for band in range(num_bands):
                        plt.subplot(2, num_bands, num_bands + band + 1)
                        plt.imshow(prediction[0, :, :, band].numpy(), cmap='gray')
                        plt.title(f'Out: {band + 1}')
                        plt.axis('off')  # Remove axes

                    plt.tight_layout()
                    plt.show()

if __name__ == "__main__":
    img = sio.loadmat('spectral_image.mat')['img']

    # Reshape the data to the appropriate shape
    (M, N, L) = img.shape
    img = img.reshape(-1, M, N, L)

    # Create a dataset that includes both images and labels
    # In this case, we use the images themselves as labels
    dataset = [img, img]

    # Now, we can use this dataset with our custom callback

    # Create an instance of the custom callback
    custom_callback = LossAndPlotCallback(dataset=dataset, plot_graph=True)

    # Define the input shape for our model
    input_shape = (M, N, L)

    # Define the encoder part of the autoencoder
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Define the decoder part of the autoencoder
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(28, (3, 3), activation='sigmoid', padding='same')(x)

    # Create the autoencoder model
    autoencoder = tf.keras.models.Model(inputs, decoded)

    # Compile the model specifying the optimizer and loss function
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Display a summary of the model architecture
    autoencoder.summary()

    # Train the model using the custom callback
    autoencoder.fit(img, img, epochs=10, callbacks=[custom_callback])
