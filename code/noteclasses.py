import os
import traceback

import cv2
import time
import numpy as np
from enum import Enum
import cmapy
import matplotlib
import notemethods as notemethods


class Side(Enum):
    FRONT = 'Front'
    BACK  = 'Back'


class Spectrum(Enum):
    RGB          = 'RGB'
    INFRARED     = 'NIR'
    HYPERSPEC    = 'HSI'
    LASERPROFILE = 'LP'
    ULTRAVIOLET  = 'UV'
    TRANSMISSIVE = 'BLT'


class FileType(Enum):
    BITMAP = 'bmp'
    RAW    = 'raw'
    HDR    = 'hdr'
    NUMPY  = 'npy'


class ImageBMP:
    def __init__(self, inputRGB, asBytes=False, inputInfo=None,
                 rotation=None, hFlip=False, vFlip=False, straighten=False):

        assert inputRGB is not None, 'Input value of None for ImageBMP'

        if asBytes:
            try:
                self.array = self.loadImageFromBytes(inputRGB, inputInfo)
            except AssertionError:
                print(traceback.format_exc())
                raise AssertionError
            except ValueError:
                print(traceback.format_exc())
                raise ValueError
            except Exception as e:
                print('Error in parsing byte stream to RGB')
                print(traceback.format_exc())
        else:
            if not os.path.exists(inputRGB):
                raise FileNotFoundError
            self.array = self.loadImageFromPath(inputRGB)

        if rotation and self.array is not None:
            self.rotate(rotation)

        if (hFlip or vFlip) and self.array is not None:
            self.flip(hFlip, vFlip)

        if straighten:
            self.straighten()

    def loadImageFromPath(self, path):
        if not os.path.exists(path):
            return None
        return cv2.imread(path)

    def loadImageFromBytes(self, byteArray, info):
        if info is None:
            assert info is not None, 'Variable inputInfo is not specified. Unable to parse from byte string'
        for e in ['height', 'width', 'spectrum']:
            assert e in info.keys(),         f'Supplied inputInfo dict has no {e} key'

        for e in ['height', 'width']:
            assert isinstance(info[e], int), f'Supplied value for {e} is not of type int'
            assert info[e] > 0,              f'Supplied value for {e} is not a valid input {info[e]}'

        assert isinstance(info['spectrum'], str), f'Supplied spectrum is not of type str'

        im = np.frombuffer(byteArray, dtype='uint8')

        channels = len(byteArray) / (info['height'] * info['width'])
        if channels == 1:
            return np.flipud(np.reshape(im, (info['height'], info['width'])))
        elif channels == 3:
            return np.flipud(np.reshape(im, (info['height'], info['width'], 3)))
        else:
            print(f'Byte array does not match expected dimensions '
                  f'{len(byteArray)} / ({info["height"]}* {info["width"]}) = {channels}')
            raise ValueError

    def show(self, resize=False):
        if resize:
            cv2.imshow('Image (Resized)', cv2.resize(self.array, (self.array.shape[0] // 2, self.array.shape[1] // 2)))
        else:
            cv2.imshow('Image', self.array)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def rotate(self, rot):
        if rot not in [0, 1, 2, 90, 180, 270]:
            print(f'Invalid rotation value, choose 90, 180, or 270. (The input was {rot})')
            raise ValueError
        if rot == 0 or rot == 90:
            self.array = cv2.rotate(self.array, 0)
        if rot == 1 or rot == 180:
            self.array = cv2.rotate(self.array, 1)
        if rot == 2 or rot == 270:
            self.array = cv2.rotate(self.array, 2)

    def flip(self, horizontal=False, vertical=False):
        if horizontal:
            self.array = np.fliplr(self.array)
        if vertical:
            self.array = np.flipud(self.array)

    def straighten(self):
        self.array = notemethods.straightenRGBIR(self.array)

class ImageHyperSpec:
    def __init__(self, input1, input2, asBytes=False, rotation=None,
                 hFlip=False, vFlip=False, straighten=False):
        if asBytes:
            self.info        = input1
            self.array       = input2
            self.array       = self.convertByteStringToArray(input2, input1)
        else:
            self.info        = self.readHeaderDict(input1)
            self.array       = self.readData(input2)

        if straighten and self.array is not None:
            self.cropAndStraighten()

        if rotation and self.array is not None:
            self.rotate(rotation)

        if (hFlip or vFlip) and self.array is not None:
            self.flip(hFlip, vFlip)

    def readHeaderDict(self, pathHDR):
        with open(pathHDR, 'r') as hf:
            hdrFile = hf.read()
        return self.parseHDRToDict(hdrFile)

    def readData(self, pathRAW):
        with open(pathRAW, 'rb') as df:
            rawFile = df.read()
        return self.convertByteStringToArray(rawFile, self.info)

    def rotate(self, rot):
        if rot not in [0, 1, 2, 90, 180, 270]:
            print(f'Invalid rotation value, choose 90, 180, or 270. (The input was {rot})')
            raise ValueError
        if rot == 0 or rot == 90:
            self.array = np.rot90(self.array, 1, axes=(1,0))
        if rot == 1 or rot == 180:
            self.array = np.rot90(self.array, 2, axes=(1,0))
        if rot == 2 or rot == 270:
            self.array = np.rot90(self.array, 3, axes=(1,0))

    def flip(self, horizontal=False, vertical=False):
        if horizontal:
            self.array = np.fliplr(self.array)
        if vertical:
            self.array = np.flipud(self.array)

    def parseHDRToDict(self, hdrString):
        if hdrString == 'GENESYS':
            return {}
        splitStr = hdrString.replace(',\n', ',').split('\n')
        headerDict = {}
        for idx, row in enumerate(splitStr):
            if 'samples' == row[:7]:
                headerDict['samples'] = int(row.split('=')[1].replace(' ', ''))
            if 'bands' == row[:5]:
                headerDict['bands'] = int(row.split('=')[1].replace(' ', ''))
            if 'lines' == row[:5]:
                headerDict['lines'] = int(row.split('=')[1].replace(' ', ''))
            if 'wavelength' in row.lower():
                headerDict['wavelengths'] = [float(w) for w in splitStr[idx+1].split(',') if w != '}']
        return headerDict

    def convertByteStringToArray(self, bString, hDict):
        hDict = checkForhDictContents(hDict)
        arr = np.frombuffer(bString, dtype='<u2')
        try:
            arr = np.reshape(arr, (hDict['lines'], hDict['bands'], hDict['samples']))
            arr = np.transpose(arr, axes=(0, 2, 1))
        except ValueError:
            print('WARNING: The collected byte string could not be reshaped. Diagnosing...')
            arr = self.diagnoseConversionError(bString, hDict)
        if arr is not None:
            self.status = True
        return arr

    def diagnoseConversionError(self, bString, hDict):
        expectedSize  = hDict['lines'] * hDict['bands'] * hDict['samples'] * 2
        actualSize    = len(bString)
        diff          = expectedSize - actualSize
        if not diff:
            print('The file is corrupted and could not be recovered')
            return None
        bString += b'0' * diff
        arr = np.frombuffer(bString, dtype='<u2')
        arr = np.reshape(arr, (hDict['lines'], hDict['bands'], hDict['samples']))
        arr = np.transpose(arr, axes=(0, 2, 1))
        return arr

    def viewSample(self, band):
        im = np.array(cv2.normalize(self.array[:,:,band], None, alpha=0, beta=1,
                                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        wavelength = self.info['wavelengths'][band]
        cv2.imshow(f'sampleband {wavelength}', im)

    def getBand(self, band, norm=True):
        im = self.array[:, : , band]
        if norm:
            im = np.array(cv2.normalize(im, None, alpha=0, beta=1,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        wavelength = self.info['wavelengths'][band]
        return im, wavelength

    def straighten(self):
        self.array = notemethods.straightenHS(self.array)

    def write(self, path):
        headerPath = f'{path}.hdr'
        rawPath    = f'{path}.raw'

        writeDir = path.replace(path.split('/')[-1], '')
        if not os.path.exists(writeDir):
            print('The write directory does not exist!')
            raise FileNotFoundError

        with open(headerPath, 'w+') as wf:
            wf.write('GENESYS')

        with open(rawPath, 'wb+') as wf:
            wf.write(self.array.tostring())


class ImageLP:
    def __init__(self, data, asBytes=False, inputInfo=None, preprocess=False):
        self.inputType = 'Bytes' if asBytes else 'File'
        if asBytes:
            self.array = self.loadDataFromBinaryString(data, inputInfo)
        else:
            self.array = np.load(data)

        if preprocess:
            self.prepareData()
            self.standardiseArray()

    def loadDataFromBinaryString(self, data, inputInfo):
        if inputInfo:
            return np.frombuffer(data, dtype='float').reshape(inputInfo['height'], inputInfo['width'])
        return np.frombuffer(data, dtype='float').reshape(-1, 2)

    def prepareData(self, inputInfo=None, drop=True):
        resolution = 2048
        if inputInfo is not None:
            resolution = inputInfo['resolution']
        # Generate Y coordinates
        numLines = self.array.shape[0] // resolution

        zeroArr = np.zeros((resolution))
        newCol  = np.array([zeroArr + i for i in range(numLines)]).reshape(-1, 1)
        self.array = np.hstack((self.array, newCol))

        # Drop timestamp data in last 4 rows
        x = np.full((resolution), True)
        x[-4:] = False
        mask = np.tile(x, numLines)
        self.array = self.array[mask == 1][:, [0, 2, 1]]

        # Drop invalid (0) z data points
        if drop:
            self.array = self.array[self.array[:, 2] != 0]

    def standardiseArray(self):
        self.array = notemethods.arrayToStandardUnits(self.array, 2, 250, self.array.shape[0], 7000)

    def depthDifferential(self, perc):
        arr   = np.absolute(np.diff(self.array[:,2]))
        bound = np.std(arr)

        ub  = np.percentile(arr, 50 + perc) + bound
        arr = np.where(arr > ub, ub, arr)

        lb  = np.percentile(arr, 50 - perc) - bound
        arr = np.where(arr < lb, lb, arr)

        self.array[:,2] = np.append(arr, [0])

    def buildPointCloud(self, scaleMin, scaleMax, view=False):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.array)
        if view:
            o3d.visualization.draw_geometries([pcd])
        return pcd

    def flatten(self):
        self.array = notemethods.flattenPointCloud(self.array)

    def projectTo2D(self, straighten=False, asmicrons=False, view=False, normalise=False):
        arr2D, x, y = notemethods.pointCloudToArray(self.array, 10)

        if straighten:
            arr2D = notemethods.straightenLP(arr2D)
        if asmicrons:
            arr2D = arr2D * 10000  # Convert from cm to micron
        if view:
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt
            plt.imshow(arr2D, cmap=cm.nipy_spectral)
            plt.axis('off')
            cbar = plt.colorbar(orientation='horizontal')
            cbar.set_label('Depth (microns)')
            plt.show()
        if normalise:
            arr2D = np.array(cv2.normalize(arr2D, None, alpha=0, beta=1,
                                        norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

        return arr2D

    def mapToColor(self, color='viridis'):
        if len(self.array.shape) > 2:
            print('ERROR: The laser profile has not yet been converted to a 2D representation')
            raise TypeError
        im = np.uint8(255 * np.array(cv2.normalize(self.array, None, alpha=0, beta=1,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)))
        return cv2.applyColorMap(im, cmapy.cmap(color))

    def rotate(self, rot):
        if rot not in [0, 1, 2, 90, 180, 270]:
            print(f'Invalid rotation value, choose 90, 180, or 270. (The input was {rot})')
            raise ValueError
        if rot == 0 or rot == 90:
            self.array = np.rot90(self.array, 1, axes=(1,0))
        if rot == 1 or rot == 180:
            self.array = np.rot90(self.array, 2, axes=(1,0))
        if rot == 2 or rot == 270:
            self.array = np.rot90(self.array, 3, axes=(1,0))

def checkForhDictContents(hDict):
    if 'samples' not in hDict:
        hDict['samples'] = 1024
    if 'bands' not in hDict:
        hDict['bands']   = 224
    if 'lines' not in hDict:
        hDict['lines']   = 370
    if 'wavelengths' not in hDict:
        hDict['wavelengths'] = [i for i in range(0, 224)]
    return hDict
