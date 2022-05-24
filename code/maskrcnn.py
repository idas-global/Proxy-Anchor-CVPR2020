import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import skimage.io
import tensorflow as tf
import numpy as np
import mrcnn.visualize as visualize
import mrcnn.model as modellib
import pandas as pd

from scipy import stats
from skimage.transform import resize
from skimage import img_as_bool
from mrcnn.config import Config

class MaskRCNN:
    def __init__(self):
        self.reqWidth  = 1024
        self.reqHeight = 512
        self.config = Config()

        dataPath = './mrcnn'
        modelPath = f'{dataPath}/mask_rcnn.h5'
        classPath = f'{dataPath}/class_labels.csv'

        # Get label data names
        self.labeldf = pd.read_csv(classPath, header=None)
        self.labeldf.columns = ['labelstr']
        self.labels  = list(self.labeldf['labelstr'])

        self.model = modellib.MaskRCNN(mode="inference", config=self.config, model_dir="./temp")
        self.model.load_weights(modelPath, by_name=True)

        self.denomModel       = tf.keras.models.load_model(f'{dataPath}/denom.h5')
        self.genuineModel     = tf.keras.models.load_model(f'{dataPath}/isGenuine.h5')

        # Force model to GPU
        self.detect(np.zeros((1024, 512, 3)).astype(np.uint8))

    def parseFeatures(self, res, labels, scale):
        resultsList = []
        for r in range(len(res['class_ids'])):
            roi       = res['rois'][r]
            classID   = res['class_ids'][r]

            resultsList.append({
                                'roi'       : roi,
                                'classID'   : classID,
                                'className' : labels[classID],
                                'score'     : res['scores'][r],
                                'mask'      : res['masks'][roi[0]:roi[2], roi[1]:roi[3], r]
                                })

        #resultsListScaled = self.scaleFix(resultsList, scale)
        return pd.DataFrame(resultsList)

    def scaleFix(self, results, scale):
        for f in results:
            f['roi'] = [int(f['roi'][0] * scale[0]),
                        int(f['roi'][1] * scale[1]),
                        int(f['roi'][2] * scale[0]),
                        int(f['roi'][3] * scale[1])]

            f['mask'] = img_as_bool(resize(f['mask'], (f['roi'][2] - f['roi'][0],
                                                       f['roi'][3] - f['roi'][1])))

        return results

    def detect(self, image, determineOrientation=False, view=False):
        image, scale = self.resize(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.model.detect([image])[0]

        if view:
            self.visualiseFeatures(image, result, self.labels)

        dfFeatures = self.parseFeatures(result, self.labels, scale)

        if determineOrientation:
            # Determine Note Side
            orientation = dfFeatures.join(self.labeldf, on='classID')
            side = orientation['side'].mode()[0]

            # Determine Note Rotation
            orientation['detectedloc'] = dfFeatures['roi'].apply(
                lambda l: 'left' if ((l[3] - l[1]) // 2 + l[1]) < (self.reqWidth // 2) else 'right')
            orientation = orientation[orientation['expectedloc'] != 'both']
            correctloc = np.where(orientation['expectedloc'] != orientation['detectedloc'], True, False)

            metaDict = {'rotated'  : bool(stats.mode(correctloc)[0][0]),
                        'side'     : side,
                        'height'   : self.reqHeight,
                        'width'    : self.reqWidth,
                        }

            return dfFeatures, metaDict

        return dfFeatures

    def visualiseFeatures(self, image, r, classNames):
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    classNames, r['scores'])

    def resize(self, image):
        originalWidth  = image.shape[1]
        originalHeight = image.shape[0]
        imResize = cv2.resize(image, (self.reqWidth, self.reqHeight))
        scaleWidth = originalWidth / self.reqWidth
        scaleHeight = originalHeight / self.reqHeight
        return imResize, (scaleHeight, scaleWidth)


    # LEGACY FUNCTION
    # def orient(self, image):
    #     im = cv2.resize(image, (self.orientWidth, self.orientHeight))
    #     im_rgb = im
    #     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #
    #     # Returns 0 if orientation note required, 1 if required
    #     orientation = self.orientationModel.predict(np.array([np.expand_dims(im, -1)]))[0].argmax()
    #
    #     if orientation:
    #         im = cv2.rotate(im, 1)
    #
    #     side = self.sideModel.predict(np.array([np.expand_dims(im, -1)]))[0].argmax()
    #     denomPrediction = None
    #     genuinePrediction = None
    #     if side:
    #         denomPrediction   = self.predictDenom(im_rgb)
    #         genuinePrediction = self.predictGenuine(im_rgb)
    #
    #     return {'flipped' : True if orientation else False,
    #             'side'    : 'back' if side else 'front',
    #             'height'  : self.reqHeight,
    #             'width'   : self.reqWidth,
    #             'denom'   : denomPrediction,
    #             'isGenuine': genuinePrediction
    #             }

    def predictDenom(self, note):
        denomLookup = {0: '10',
                       1: '20',
                       2: '50',
                       3: '100'}

        class_idx = self.denomModel.predict(np.expand_dims(note, axis=0))
        return denomLookup[np.argmax(class_idx)]

    def predictGenuine(self, note):
        return int(np.argmax(self.genuineModel.predict(np.expand_dims(note, axis=0))))

def test():
    # Load a random image from the images folder
    imagePath = './data/mrcnn/note.bmp'
    image = skimage.io.imread(imagePath)

    maskrcnn = MaskRCNN()
    ret = maskrcnn.detect(image, view=True)
    print(ret)
    print(maskrcnn.orient(image))

if __name__ == '__main__':
    test()
