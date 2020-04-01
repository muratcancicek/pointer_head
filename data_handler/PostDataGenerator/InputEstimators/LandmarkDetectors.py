# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
from .InputEstimatorABC import InputEstimatorABC
from .FaceDetectors import CVFaceDetector
from ...Paths import YinsFacialLandmarkDetector_tf_model_path
import cv2, numpy as np, tensorflow as tf
from abc import abstractmethod

class FacialLandmarkDetectorABC(InputEstimatorABC):

    def __init__(self, faceDetector = None, inputLandmarkIndex = 30, *args, **kwargs):
        self._facialLandmarks = np.zeros((1,2))
        self._inputLandmarkIndex = inputLandmarkIndex
        self._inputLandmark = np.zeros((3,))
        self._faceDetector = faceDetector
        super().__init__(*args, **kwargs)
        
    @classmethod
    @abstractmethod
    def detectFacialLandmarks(self, frame):
        raise NotImplementedError
    
    @property
    def facialLandmarks(self):
            return self._facialLandmarks
        
    def findInputLandmarkLocation(self, frame):
        landmarks = self.detectFacialLandmarks(frame)
        if len(landmarks) > self._inputLandmarkIndex:
            self._facialLandmarks = landmarks
            self._inputLandmark[:2] = self._facialLandmarks[self._inputLandmarkIndex]
        return self._inputLandmark

    def estimateInputValues(self, frame):
        return self.findInputLandmarkLocation(frame)
    
    def findInputLandmarkLocationWithAnnotations(self, frame):
        return self.findInputLandmarkLocation(frame), self._faceDetector.getProjectionPoints(), self._facialLandmarks

    def estimateInputValuesWithAnnotations(self, frame):
        return self.findInputLandmarkLocationWithAnnotations(frame)

    @property
    def inputValues(self):
        return self._inputLandmark

    def returns3D(self):
        return False

# The code is derived from the following repository:
# https://github.com/yinguobing/head-pose-estimation

class LandmarkDetector(FacialLandmarkDetectorABC):

    @staticmethod
    def loadTFGraph(tf_model_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(tf_model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def __init__(self, faceDetector = None, inputLandmarkIndex = 30, tf_model_path = None, *args, **kwargs):
        if faceDetector == None:
            faceDetector = CVFaceDetector(squaringFaceBox = True)
        super().__init__(faceDetector, inputLandmarkIndex, *args, **kwargs)
        if tf_model_path == None:
            tf_model_path = YinsFacialLandmarkDetector_tf_model_path

        self.__graph = self.loadTFGraph(tf_model_path)
        self.__sess = tf.Session(graph = self.__graph)
        self.__cnn_input_size = 128

    def __detectFaceImage(self, frame):
        faceBox = self._faceDetector.detectFaceBox(frame)
        if faceBox == None:
            squaredFaceImage = frame
        else:
            squaredFaceImage = faceBox.getSquaredFaceImageFromFrame(frame)
        squaredFaceImage = cv2.resize(squaredFaceImage, (self.__cnn_input_size, self.__cnn_input_size))
        return cv2.cvtColor(squaredFaceImage, cv2.COLOR_BGR2RGB)

    def detectFacialLandmarks(self, frame):
        faceImage = self.__detectFaceImage(frame)

        logits_tensor = self.__graph.get_tensor_by_name('logits/BiasAdd:0')
        predictions = self.__sess.run(logits_tensor, feed_dict={'input_image_tensor:0': faceImage})

        marks = np.array(predictions).flatten()
        marks = np.reshape(marks, (-1, 2))
        #marks = marks * frame.shape[:2]
        #print(marks.shape)
        if self._faceDetector.faceBox != None:
            marks[:, 0] *= self._faceDetector.faceBox.width
            marks[:, 1] *= self._faceDetector.faceBox.height
            marks[:, 0] += self._faceDetector.faceBox.left
            marks[:, 1] += self._faceDetector.faceBox.top
        self._facialLandmarks = marks.astype(int)
        return self._facialLandmarks