# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from PostDataGenerator.InputEstimators.PoseEstimators import HeadPoseEstimatorABC, HeadGazer 
from PostDataGenerator.InputEstimators.LandmarkDetectors import FacialLandmarkDetectorABC, LandmarkDetector
from PostDataGenerator.InputEstimators.FaceDetectors import FaceBox, FaceDetectorABC, CVFaceDetector

from abc import ABC, abstractmethod
import numpy as np

class Boundary(object):

    def __init__(self, minX = None, maxX = None, minY = None, maxY = None, 
                 minZ = None, maxZ = None, *args, **kwargs):
        minX = float('-inf') if minX == None else float(minX)
        maxX = float('inf') if maxX == None else float(maxX)
        self.__xRange = float('inf') if maxX == None else abs(maxX - minX)
        if minX > maxX: t = minX; minX = maxX; maxX = minX
        minY = float('-inf') if minY == None else float(minY)
        maxY = float('inf') if maxY == None else float(maxY)
        self.__yRange = float('inf') if maxY == None else abs(maxY - minY)
        if minY > maxY: t = minY; minY = maxY; maxY = minY
        minZ = float('-inf') if minZ == None else float(minZ)
        maxZ = float('inf') if maxZ == None else float(maxZ)
        self.__zRange = float('inf') if maxZ == None else abs(maxZ - minZ)
        if minZ > maxZ: t = minZ; minZ = maxZ; maxZ = minZ
        self.__minX, self.__minY, self.__minZ = minX, minY, minZ
        self.__maxX, self.__maxY, self.__maxZ = maxX, maxY, maxZ
        super().__init__()

    def isInRanges(self, x = None, y = None, z = None):
        if x == None: xIn = True
        else: xIn = self.__minX < x and x < self.__maxX
        if y == None: yIn = True
        else: yIn = self.__minY < y and y < self.__maxY
        if z == None: zIn = True
        else: zIn = self.__minZ < z and z < self.__maxZ
        return xIn and yIn and zIn 

    def isIn(self, point):
        if len(point) == 2:
            z = self.__minZ + self.__zRange/2  
        else: 
            z = point[2]
        return self.isInRanges(point[0], point[1], z)

    def getRanges(self):
        return self.__xRange, self.__yRange, self.__zRange

    def keepInside(self, point):
        if point[0] < self.__minX:
            point[0] = self.__minX + 1
        elif self.__maxX < point[0]:
            point[0] = self.__maxX - 1
        if point[1] < self.__minY:
            point[1] = self.__minY + 1
        elif self.__maxY < point[1]:
            point[1] = self.__maxY - 1
        if len(point) == 2:
            return point
        if point[2] < self.__minZ:
            point[2] = self.__minZ + 1
        elif self.__maxZ < point[2]:
            point[2] = self.__maxZ - 1
        return point

    def flipXYAxes(self):
        minX, minY = self.__minX, self.__minY
        self.__minX, self.__minY = minY, minX

        maxX, maxY = self.__maxX, self.__maxY
        self.__maxX, self.__maxY = maxY, maxX

        t = self.__xRange
        self.__yRange = self.__xRange
        self.__yRange = t
        return self

    def getAbsVolume(self, point):
        if len(point) == 2:
            return point - (self.__minX, self.__minY)
        else:
            return point - (self.__minX, self.__minY, self.__minZ)

    def getVolumeRatio(self, point):
        xR = 1 if self.__xRange == float('inf') else self.__xRange
        yR = 1 if self.__yRange == float('inf') else self.__yRange
        xRatio = 0 if xR == 0 else point[0] / xR
        yRatio = 0 if yR == 0 else point[1] / yR
        if len(point) == 2:
            return np.array((xRatio, yRatio))
        else:
            zR = 1 if self.__zRange == float('inf') else self.__zRange
            zRatio = 0 if zR == 0 else point[2] / zR
            return np.array((xRatio, yRatio, zRatio))

    def getVolumeAbsRatio(self, point):
        point1 = self.getAbsVolume(point)
        return self.getVolumeRatio(point1)

    def __str__(self):
        return 

class MappingABC(ABC):

    def __init__(self, inputEstimator, outputBoundaries, *args, **kwargs):
        self._inputEstimator = inputEstimator
        self._initializeInputCalculator()
        self._inputValues = np.zeros((3,))
        self._outputValues = np.zeros((2,))
        self._outputBoundaries = outputBoundaries
        super().__init__()
    
    def _calculateInputValuesFromFaceBox(self):
        currentFaceBox = self._inputEstimator.faceBox
        left, right = currentFaceBox.left, currentFaceBox.right
        top, bottom = currentFaceBox.top, currentFaceBox.bottom
        if self._inputBoundaries == None:
                self._inputBoundaries = Boundary(left, right, top, bottom)
                self._faceBoxForInput = currentFaceBox
        x, y = self._inputValues[:2]
        update = False
        if not self._inputBoundaries.isInRanges(x = x):
            update = True
            if x - self._faceBoxForInput.location[0] > 0:
                left, right = x - (right - left), x
            else:
                left, right = x, x + (right - left)
            top = self._faceBoxForInput.top
            bottom = self._faceBoxForInput.bottom
        if not self._inputBoundaries.isInRanges(y = y):
            update = True
            if y - self._faceBoxForInput.location[1] > 0:
                top, bottom = y - (bottom - top), y
            else:
                top, bottom = y, y + (bottom - top)
            left = self._faceBoxForInput.left
            right = self._faceBoxForInput.right
        if update:
            self._inputBoundaries = Boundary(left, right, top, bottom)
            self._faceBoxForInput = FaceBox(int(left), int(top),
                                           int(right), int(bottom))
        self._pPoints = self._faceBoxForInput.getProjectionPoints()
        return self._inputValues
        
    def _calculateInputValuesFromNose(self):
        minX, maxX = self._Landmarks[49, 0], self._Landmarks[53, 0]
        minY = (self._Landmarks[1, 1] + self._Landmarks[15, 1])/2
        maxY = (self._Landmarks[4, 1] + self._Landmarks[12, 1])/2
        self._inputValues[:2] = self._inputValues[:2] - (minX, minY)
        minX, maxX, minY, maxY =  0, maxX - minX, 0, maxY  - minY
        self._inputBoundaries = Boundary(minX, maxX, minY, maxY)
        return self._inputValues
    
    def _calculateInputValuesFromHeadPose(self):
        return self._inputValues

    def _calculateInputValuesFromHeadGaze(self):
        return self._inputValues

    def _recalculateInputValues(self):
        raise NotImplementedError
           
    def _initializeInputCalculator(self):
        self._outputDependsAnnotations = False
        self._inputBoundaries = Boundary()
        if isinstance(self._inputEstimator, FaceDetectorABC):
            self._recalculateInputValues = self._calculateInputValuesFromFaceBox
            self._inputBoundaries = None
        elif isinstance(self._inputEstimator, FacialLandmarkDetectorABC):
            self._recalculateInputValues = self._calculateInputValuesFromNose
            self._outputDependsAnnotations = True
        elif isinstance(self._inputEstimator, HeadGazer):
            width, height = self._inputEstimator.getGazingFrameDimensions()
            self._inputBoundaries = Boundary(0, width, 0, height)
            self._recalculateInputValues = self._calculateInputValuesFromHeadGaze
        elif isinstance(self._inputEstimator, HeadPoseEstimatorABC):
            self._inputBoundaries = Boundary(10, 30, -10, 10)
            self._recalculateInputValues = self._calculateInputValuesFromHeadPose

    def _estimateInput(self, frame):
        if self._outputDependsAnnotations:
            annos = self._inputEstimator.estimateInputValuesWithAnnotations(frame)
            self._inputValues, self._pPoints, self._Landmarks = annos
        else:
            self._inputValues = self._inputEstimator.estimateInputValues(frame)
        return self._inputValues 

    @abstractmethod
    def _calculate(self):
        raise NotImplementedError
        
    def _updateOutputValues(self):
        self._calculate()
        self._outputValues = self._outputBoundaries.keepInside(self._outputValues)
        return self._outputValues

    def calculateOutputValues(self, frame):
        self._estimateInput(frame)
        self._recalculateInputValues()
        return self._updateOutputValues()

    def calculateOutputValuesWithAnnotations(self, frame):
        self._outputDependsAnnotations = True
        self.calculateOutputValues(frame)
        return self._outputValues, self._inputValues, self._pPoints, self._Landmarks

    @property
    def inputValues(self):
        return self._inputValues

    @property
    def outputValues(self):
        return self._outputValues

    def getEstimator(self):
        return self._inputEstimator

    def getOutputBoundaries(self):
        return self._outputBoundaries

    def getInputBoundaries(self):
        return self._inputBoundaries

class StaticMapping(MappingABC):
        
    def _calculate(self):
        #inputRanges = self._inputBoundaries.getRanges()
        #outputRanges = self._outputBoundaries.getRanges()
        #ratios = self._inputBoundaries.getVolumeAbsRatio(self._inputValues)
        #if isinstance(self._inputEstimator, HeadPoseEstimatorABC) and \
        #           not isinstance(self._inputEstimator, HeadGazer):
        #    t = ratios[0]; ratios[0] = ratios[1]; ratios[1] = t
        #i = self._outputValues.shape[0]
        self._outputValues = self._inputValues #ratios[:i] * outputRanges[:i]
        return self._inputValues
    
class DynamicMapping(MappingABC):
    def __init__(self, inputEstimator, outputBoundaries, xSpeed = 1, ySpeed = 1,
                acceleration = 2, smoothness = 30, motionThreshold = 4, *args, **kwargs):
        super().__init__(inputEstimator, outputBoundaries, *args, **kwargs)
        if smoothness < 2: smoothness = 2
        self._inputValueQueue = None
        self._outputValueQueue = None
        self._speed = np.array((xSpeed, ySpeed))
        self._acceleration = acceleration
        self._smoothness = smoothness
        self._motionThreshold = motionThreshold
        
    def _initializeQueues(self):
        if not self._inputValueQueue is None: return
        self._inputValueQueue = np.zeros((self._smoothness, 3))
        self._outputValueQueue = np.zeros((self._smoothness, 2))
        for i in range(self._inputValueQueue.shape[0]):
           self._inputValueQueue[i] = self._inputValues
        outputRanges = self._outputBoundaries.getRanges()[:2]
        for i in range(self._outputValueQueue.shape[0]):
           self._outputValueQueue[i] = outputRanges
        self._outputValueQueue = self._outputValueQueue/2
        return

    def _calculate(self):
        if isinstance(self._inputEstimator, HeadPoseEstimatorABC):
            t = self._inputValues[0]
            self._inputValues[0] = self._inputValues[1] 
            self._inputValues[1] = t
        self._initializeQueues()
        inputRanges = self._inputBoundaries.getRanges()
        outputRanges = self._outputBoundaries.getRanges()

        self._inputValueQueue[:-1, :] = self._inputValueQueue[1:, :]
        self._inputValueQueue[-1, :] = self._inputValues
        self._inputValues = self._inputValueQueue.mean(axis = 0)

        direction = self._inputValues - self._inputValueQueue[-2, :] 
        direction = (direction[:2]/inputRanges[:2] * self._speed)*outputRanges[:2]

        self._outputValueQueue[:-1, :] = self._outputValueQueue[1:, :]
        self._outputValueQueue[-1, :] = self._outputValueQueue[-2, :] - direction
        self._outputValues = self._outputValueQueue.mean(axis = 0)
        
        if isinstance(self._inputEstimator, HeadPoseEstimatorABC): 
            self._outputValues[1] -= int(outputRanges[1]/3) # manually lifting the pointer

        return self._outputValues