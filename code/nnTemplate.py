import os
import cv2
import shutil
import random
import numpy as np
import datetime
from tensorflow import keras
import tensorflow.keras.callbacks as tkc
import tensorflow.keras.layers as tkl
import tensorflow.keras.models as tkm
import tensorflow.keras.optimizers as tko
from tensorflow.keras import backend as K
import tfcallbacks

def loadImage(file_path):
    im = np.expand_dims(cv2.imread(file_path, cv2.IMREAD_GRAYSCALE), axis=-1)
    return im

def loadImageRGB(file_path):
    im = cv2.imread(file_path)
    return im

class Inline_Generator(keras.utils.Sequence) :
    def __init__(self, inputFilePaths, outputFilePaths, batchSize, color=True) :
        self.imageFileNames = inputFilePaths
        self.labelFileNames = outputFilePaths
        self.batchSize = batchSize
        self.imLoad = loadImageRGB if color else loadImage

    def __len__(self) :
        return (np.ceil(len(self.imageFileNames) / float(self.batchSize))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.imageFileNames[idx * self.batchSize: (idx+1) * self.batchSize]
        batch_y = self.labelFileNames[idx * self.batchSize: (idx+1) * self.batchSize]

        return np.array([self.imLoad(f) for f in batch_x]) / 255.0, \
               np.array([self.imLoad(f) for f in batch_y]) / 255.0

class AutoEncoder():
    def __init__(self, imShape, filters=(64, 32), latentDim=16):
        self.filters = filters
        self.imShape = imShape
        self.latent  = latentDim

    def build(self):
        inputs = tkl.Input(shape=self.imShape)
        x = inputs
        for f in self.filters:
            x = tkl.Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = tkl.LeakyReLU(alpha=0.2)(x)
            x = tkl.BatchNormalization(axis=-1)(x)

        volumeSize = K.int_shape(x)
        x = tkl.Flatten()(x)
        latent  = tkl.Dense(self.latent)(x)
        encoder = tkm.Model(inputs, latent, name="encoder")

        latentInputs = tkl.Input(shape=(self.latent,))
        x = tkl.Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = tkl.Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        for f in self.filters[::-1]:
            x = tkl.Conv2DTranspose(f, (3, 3), strides=2,padding="same")(x)
            x = tkl.LeakyReLU(alpha=0.2)(x)
            x = tkl.BatchNormalization(axis=-1)(x)

        x = tkl.Conv2DTranspose(self.imShape[2], (3, 3), padding="same")(x)
        outputs = tkl.Activation("sigmoid")(x)
        decoder = tkm.Model(latentInputs, outputs, name="decoder")
        autoencoder = tkm.Model(inputs, decoder(encoder(inputs)), name="autoencoder")

        return encoder, decoder, autoencoder

def getTrainingData(noisyPath, cleanPath):
    fNames     = os.listdir(noisyPath)
    random.shuffle(fNames)
    noisyFiles = [f'{noisyPath}/{f}' for f in fNames]
    cleanFiles = [f'{cleanPath}/{f}' for f in fNames]
    return noisyFiles, cleanFiles

def splitTrainTestData(noisyFiles, cleanFiles):
    split = len(noisyFiles) // 5
    return noisyFiles[split:], cleanFiles[split:], noisyFiles[:split], cleanFiles[:split]

def getShape(im):
    return cv2.imread(im).shape

def buildAndTrain(noisyPath, cleanPath, epochs=50, batchSize=64):
    if os.path.exists('logs'): shutil.rmtree('logs')
    noisyFiles, cleanFiles = getTrainingData(noisyPath, cleanPath)
    trainX, trainY, testX, testY = splitTrainTestData(noisyFiles, cleanFiles)

    trainBatchGenerator = Inline_Generator(trainX, trainY, batchSize)
    testBatchGenerator  = Inline_Generator(testX, testY, batchSize)

    inputSize = getShape(trainX[0])

    encoder, decoder, autoencoder = AutoEncoder(inputSize, filters=(512, 256, 128), latentDim=256).build()
    autoencoder.compile(loss="mse", optimizer=tko.Adam(lr=0.01))

    # Logs and callbacks
    logDir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #earlyStopper = tkc.EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=50, verbose=1, mode='min')
    modelCheckpt = tkc.ModelCheckpoint('checkpoint.h5', save_best_only=True)
    tensorBoard  = tkc.TensorBoard(log_dir=logDir, histogram_freq=1)
    imageTesting = tfcallbacks.CustomCallback(autoencoder, testBatchGenerator.__getitem__(2)[0])

    H = autoencoder.fit(x=trainBatchGenerator,
                        validation_data=testBatchGenerator,
                        epochs=epochs,
                        batch_size=batchSize,
                        callbacks=[modelCheckpt, tensorBoard, imageTesting])

    autoencoder.save('model.h5')

def main():
    dataDir   = './datasets/shapes'
    noisyPath = dataDir + '/original'
    cleanPath = dataDir + '/original'
    epochs    = 200
    batchSize = 64
    buildAndTrain(noisyPath, cleanPath, epochs, batchSize)

if __name__ == '__main__':
    main()