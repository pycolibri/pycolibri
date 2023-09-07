import tensorflow as tf
import matplotlib.pyplot as plt


class LossAndPredictionCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, num_samples=5, plot_graph=False):
        super(LossAndPredictionCallback, self).__init__()
        self.dataset = dataset
        self.num_samples = num_samples
        self.plot_graph = plot_graph

    def on_epoch_end(self, epoch, logs=None):
        # Obtén muestras aleatorias del conjunto de datos
        sample_indices = tf.random.uniform(
            (self.num_samples,), maxval=len(self.dataset), dtype=tf.int32
        )
        sampled_images, sampled_labels = [], []
        for index in sample_indices.numpy():
            image, label = self.dataset[index]
            sampled_images.append(image)
            sampled_labels.append(label)

        sampled_images = tf.stack(sampled_images)
        sampled_labels = tf.stack(sampled_labels)

        # Realiza predicciones en las muestras seleccionadas
        predictions = self.model.predict(sampled_images)

        # Dibuja las imágenes de Ground Truth y las predicciones
        if self.plot_graph:
            plt.figure(figsize=(12, 6))
            for i in range(self.num_samples):
                plt.subplot(self.num_samples, 2, 2 * i + 1)
                plt.imshow(sampled_images[i].numpy().squeeze(), cmap='gray')
                plt.title('Ground Truth')

                plt.subplot(self.num_samples, 2, 2 * i + 2)
                plt.imshow(predictions[i].squeeze(), cmap='gray')
                plt.title('Predicción')

            plt.tight_layout()
            plt.show()

        # Reporta la pérdida en el registro
        print(f'Epoch {epoch + 1} - Loss: {logs["loss"]:.4f}')
