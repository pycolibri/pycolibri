import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio

class LossAndEncoderOutputCallback(tf.keras.callbacks.Callback):

    def __init__(self, dataset, num_samples=1, plot_graph=False):
        """
        Callback para mostrar pérdida y predicciones durante el entrenamiento.

        Parameters
        ----------
        dataset : tf.data.Dataset
            El conjunto de datos de entrenamiento.
        num_samples : int, optional
            El número de muestras aleatorias para mostrar, por defecto es 5.
        plot_graph : bool, optional
            Indica si se deben trazar las imágenes de Ground Truth y las predicciones, por defecto es False.

        """

        super(LossAndEncoderOutputCallback, self).__init__()
        self.dataset = dataset
        self.num_samples = num_samples
        self.plot_graph = plot_graph

    def on_epoch_end(self, epoch, logs=None):
        """
        Llamado al final de cada época para mostrar pérdida y predicciones.

        Parameters
        ----------
        epoch : int
            Número de la época actual.
        logs : dict, optional
            Diccionario que contiene las métricas de entrenamiento, por defecto es None.

        """

        # Calcular el promedio de la pérdida en esta época
        epoch_loss = logs["loss"]
        epoch_num = epoch + 1

        # Imprimir la pérdida promedio de esta época
        print('\n\n' + f'Época {epoch_num} - Pérdida Promedio: {epoch_loss:.4f}' + '\n\n')

        # Calcular la salida del codificador para las imágenes muestreadas
        for _ in range(self.num_samples):
            index = tf.random.uniform(
                (), maxval=len(self.dataset), dtype=tf.int32
            )
            sampled_images = self.dataset[index]  # Una lista de imágenes

            # Procesar cada imagen individualmente
            for sampled_image in sampled_images:
                sampled_image = tf.reshape(sampled_image, (1, 256, 256, 28))
                encoder_output = self.model.layers[2](sampled_image)  # Cambiar al índice correcto de la capa

                # Plotear la imagen de entrada vs. la salida de la red
                if self.plot_graph:
                    num_bands = sampled_image.shape[3]  # Número de bandas en la imagen de entrada

                    # Crear una figura con subplots para cada banda de entrada y salida
                    plt.figure(figsize=(num_bands, 2))  # Aumentamos el tamaño de la figura

                    # Subplots para la entrada
                    for band in range(num_bands):
                        plt.subplot(2, num_bands, band + 1)
                        plt.imshow(sampled_image[0, :, :, band].numpy(), cmap='gray')
                        plt.title(f'In: {band + 1}')
                        plt.axis('off')  # Eliminamos los ejes

                    # Subplots para la salida
                    for band in range(num_bands):
                        plt.subplot(2, num_bands, num_bands + band + 1)
                        plt.imshow(encoder_output[0, :, :, band].numpy(), cmap='gray')
                        plt.title(f'Out: {band + 1}')
                        plt.axis('off')  # Eliminamos los ejes

                    plt.tight_layout()
                    plt.show()





img = sio.loadmat('../examples/data/spectral_image.mat')['img']

img = img.reshape(-1, 256, 256, 28)

dataset = [img, img]

custom_callback = LossAndEncoderOutputCallback(dataset=dataset, plot_graph=True)


input_shape = (256, 256, 28)

inputs = tf.keras.layers.Input(shape=input_shape)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
encoded = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
decoded = tf.keras.layers.Conv2D(28, (3, 3), activation='sigmoid', padding='same')(x)


autoencoder = tf.keras.models.Model(inputs, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

autoencoder.fit(img, img, epochs=10, callbacks=[custom_callback])
