# The code is derived from the following repository:
# https://github.com/yeephycho/tensorflow-face-detection

from .InputEstimatorABC import InputEstimatorABC
from ...Paths import TFMobileNetSSDFaceDetector_tf_model_path
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
import cv2

class FaceBox(object):
    def __init__(self, left, top, right, bottom, *args, **kwargs):
        if left < right:
            self.left = left  
            self.right = right
        else:
            self.left = right
            self.right = left  
        if top < bottom:
            self.top = top
            self.bottom = bottom
        else:
            self.top = bottom
            self.bottom = top
        self._tl_corner = (self.left, self.top)
        self._tr_corner = (self.right, self.top)
        self._bl_corner = (self.left, self.bottom)
        self._br_corner = (self.right, self.bottom)
        self.width = abs(self.right - self.left)
        self.height = abs(self.bottom - self.top)
        self.location = (self.left + self.width/2, self.top + self.height/2)
        super().__init__(*args, **kwargs)
    
    def getProjectionPoints(self):
        corners = np.array([self._tl_corner, self._tr_corner, self._br_corner, self._bl_corner])
        return corners 

    def isSquare(self):
        return self.width == self.height
        
    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]
        return [left_x, top_y, right_x, bottom_y]
        
    def move(self, offset):
        left_x = self.left + offset[0]
        top_y = self.top + offset[1]
        right_x = self.right + offset[0]
        bottom_y = self.bottom + offset[1]
        #self.__init__#= self
        return FaceBox(left_x, top_y, right_x, bottom_y) 

    def __squareFaceBox(self, f_height, f_width):
        left, top, right, bottom = self.left, self.top, self.right, self.bottom
        if left < 0: left = 0
        if right >= f_width: right = f_width-1
        if top < 0: top = 0
        if bottom >= f_height: bottom = f_height-1

        diff = self.width - self.height
        if diff == 1:
            if self.width > self.height:
                if top > 0:
                    return FaceBox(left, top - 1, right, bottom)
                else:
                    return FaceBox(left, 0, right, bottom + 1)
            else:
                if left > 0:
                    return FaceBox(left - 1, top, right, bottom)
                else:
                    return FaceBox(0, top, right + 1, bottom)

        if abs(diff) % 2 == 1:
            diff += 1 if diff > 0 else -1
        halfDiff = int(diff/2)
        if diff > 0:
            diff, halfDiff = abs(diff), abs(halfDiff)
            if top >= halfDiff and bottom < f_height - halfDiff:
                return FaceBox(left, top - halfDiff, right, bottom + halfDiff)
            elif top < halfDiff:
                return FaceBox(left, 0, right, bottom + (diff - top))
            else:
                return FaceBox(left, (top - diff) + (f_height - bottom), right, f_height)
        else:
            diff, halfDiff = abs(diff), abs(halfDiff)
            if left >= halfDiff and right < f_width - halfDiff:
                return FaceBox(left - halfDiff, top, right + halfDiff, bottom)
            elif left < halfDiff:
                return FaceBox(0, top, right + (diff - left), bottom)
            else:
                return FaceBox((left - diff) + (f_width - right), top, f_width, bottom)

    def getSquareFaceBoxOnFrame(self, frame):
        if self.isSquare():
            return self
        else:
            f_height, f_width = frame.shape[:2]
            squaredFaceBox = self.__squareFaceBox(f_height, f_width)
            return squaredFaceBox.getSquareFaceBoxOnFrame(frame)

    def getFaceImageFromFrame(self, frame):
        #print(self.top,self.bottom, self.left, self.right, '  ')
        return frame[self.top:self.bottom, self.left:self.right]

    def getSquaredFaceImageFromFrame(self, frame):
        squaredFaceBox = self.getSquareFaceBoxOnFrame(frame)
        return squaredFaceBox.getFaceImageFromFrame(frame)

class FaceDetectorABC(InputEstimatorABC):

    @abstractmethod
    def __init__(self, squaringFaceBox = False, *args, **kwargs):
        self._faceBox = None
        self._squaringFaceBox = squaringFaceBox
        self._faceLocation = np.zeros((3,))
        super().__init__(*args, **kwargs)
        
    @staticmethod
    @abstractmethod
    def _decodeFaceBox(self, detection):
        raise NotImplementedError
        
    @abstractmethod
    def _detectFaceBox(self, frame):
        raise NotImplementedError

    def detectFaceBox(self, frame):
        self._faceBox = self._detectFaceBox(frame)
        if self._faceBox != None and self._squaringFaceBox:
            self._faceBox = self._faceBox.getSquareFaceBoxOnFrame(frame)
        return self._faceBox

    def detectFaceImage(self, frame):
        self._faceBox = self.detectFaceBox(frame)
        if self._faceBox == None:
            return self._faceBox
        else:
            return self._faceBox.getFaceImageFromFrame(frame)

    def findFaceLocation(self, frame):
        self._faceBox = self.detectFaceBox(frame)
        if self._faceBox == None:
            return self._faceLocation
        else:
            self._faceLocation[0] = self._faceBox.location[0]
            self._faceLocation[1] = self._faceBox.location[1]
        self._inputValues = self._faceLocation.copy()
        return self._faceLocation

    def estimateInputValues(self, frame):
        return self.findFaceLocation(frame)

    def getProjectionPoints(self):
        if self._faceBox == None:
            return None
        return self._faceBox.getProjectionPoints()

    def findFaceLocationWithAnnotations(self, frame):
        return self.findFaceLocation(frame), self.getProjectionPoints(), [self._faceLocation[:2].astype(int)]

    def estimateInputValuesWithAnnotations(self, frame):
        return self.findFaceLocationWithAnnotations(frame)

    @property
    def faceBox(self):
        return self._faceBox
            
    @property
    def inputValues(self):
        return self._faceLocation

    def returns3D(self):
        return False

class CVFaceDetector(FaceDetectorABC):
    def __init__(self, confidence_threshold = 0.90, dnn_proto_text_path = None, dnn_model_path = None, *args, **kwargs):
        if dnn_proto_text_path == None:
            dnn_proto_text_path = 'C:/cStorage/Datasets/CV2Nets/CV2Res10SSD/deploy.prototxt'
        if dnn_model_path == None:
            dnn_model_path = 'C:/cStorage/Datasets/CV2Nets/CV2Res10SSD/res10_300x300_ssd_iter_140000.caffemodel'
        self.__detector = cv2.dnn.readNetFromCaffe(dnn_proto_text_path, dnn_model_path)
        self.__confidence_threshold = confidence_threshold  
        super().__init__(*args, **kwargs)
        
    @staticmethod
    def _decodeFaceBox(detection):
        (rows, cols, _), detection = detection
        x_left_bottom = int(detection[3] * cols)
        y_left_bottom = int(detection[4] * rows)
        x_right_top = int(detection[5] * cols)
        y_right_top = int(detection[6] * rows)
        return FaceBox(x_left_bottom, y_right_top, x_right_top, y_left_bottom)

    def _detectFaceBox(self, frame):
        confidences = []
        faceBoxDetections = []

        self.__detector.setInput(cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (0, 0, 0), False, False))
        detections = self.__detector.forward()

        for detection in detections[0, 0, :, :]:
            confidence = detection[2]
            if confidence > self.__confidence_threshold:
                confidences.append(confidence)
                faceBoxDetections.append((frame.shape, detection))

        if len(faceBoxDetections) > 0: 
            self._faceBox = self._decodeFaceBox(faceBoxDetections[0])
        return self._faceBox
        