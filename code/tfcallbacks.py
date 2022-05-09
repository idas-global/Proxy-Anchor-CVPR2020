import io
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime

def plotToImage(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

def imageAsFigure(im1, im2, im3, im4):
    figure, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(im1)
    axs[1].imshow(im2)
    axs[2].imshow(im3)
    axs[3].imshow(im4)
    for p in range(4):
        axs[p].axis('off')
    plt.tight_layout()
    return figure

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, testImages):
        """ Save params in constructor
        """
        self.model = model
        self.testImages = testImages
        self.logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    def on_epoch_end(self, epoch, logs={}):
        x = self.model.predict(self.testImages)
        idx1 = random.randint(0,20)
        idx2 = random.randint(0,20)
        imAsTensor  = plotToImage(imageAsFigure(self.testImages[idx1], x[idx1],
                                                self.testImages[idx2], x[idx2]))

        file_writer_cm = tf.summary.create_file_writer(self.logdir + '/im')

        with file_writer_cm.as_default():
            tf.summary.image(f"E{epoch} - Sample", imAsTensor, step=epoch)