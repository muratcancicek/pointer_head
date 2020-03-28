from PostDataGenerator.InputEstimators.MappingFunctions import Boundary, StaticMapping, DynamicMapping
from PostDataGenerator.InputEstimators.InputEstimationVisualizer import InputEstimationVisualizer
from PostDataGenerator.InputEstimators.Scene3DVisualizer import Scene3DVisualizer
from PostDataGenerator.InputEstimators.PoseEstimators import PoseEstimator, HeadGazer
#from PostDataGenerator.InputEstimators.LandmarkDetectors import LandmarkDetector
#from PostDataGenerator.InputEstimators.FaceDetectors import CVFaceDetector
from datetime import datetime
import numpy as np
import Paths
import cv2
import os

class PostDataGenerator(object):
    def __init__(self):
        super()
        self.__estimator = HeadGazer() # PoseEstimator() # LandmarkDetector() # CVFaceDetector()
        self.__visualizer = Scene3DVisualizer() # InputEstimationVisualizer() # 

    def openVideo(self, path):
        cap = cv2.VideoCapture(path)
        frameCount = 0
        while(True):
            ret, frame = cap.read()
            #subjFrame = cv2.flip(subjFrame, 1)
            if not ret:
                if frameCount < 1:
                    print('Something Wrong')
                break
            frameCount += 1
            yield frame
        cap.release()    
        return

    def initializeRecorder(self, id, trailName, fps = 30, dims = (1920, 1080)):
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        dir = Paths.MergedVideosFolder + ('%s%s' % (id, Paths.sep))
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok = True)
        now = str(datetime.now())[:-7].replace(':', '-').replace(' ', '_')
        recordName = trailName + '_%s_%s_with_pointer2.avi' % (id, now)
        return  cv2.VideoWriter(dir + recordName, fourcc, fps, dims)
 
    def recordSubjectVideoWithAllInputs(self, id, trailName, subjectVideoPath):
        recorder = self.initializeRecorder(id, trailName)
        for subjFrame in self.openVideo(subjectVideoPath):
            annotations = self.__estimator.estimateInputValuesWithAnnotations(subjFrame)
            inputValues, pPoints, landmarks = annotations
            k = self.__visualizer.showFrameWithAllInputs(subjFrame, pPoints, landmarks)
            recorder.write(subjFrame)
            print('\rMerging frames (%d)...' % frameCount, end = '\r')   
        recorder = None
        print('\r%s and %s have been merged.' % \
            (subjectVideoPath.split(Paths.sep)[-1], trailName), end = '\r')
        
    def playSubjectVideoWithAllInputs(self, subjectVideoPath):
        streamer = self.openVideo(subjectVideoPath)
        self.__visualizer.playSubjectVideoWithAllInputs(self.__estimator, streamer)
        return

    def playSubjectVideoWithHeadGaze(self, subjectVideoPath):
        outputSize = (1920, 1080)
        boundary = Boundary(0, outputSize[0], 0, outputSize[1])
        #self._mappingFunc = DynamicMapping(self.__estimator, boundary)
        mappingFunc = StaticMapping(self.__estimator, boundary)
        streamer = self.openVideo(subjectVideoPath)
        self.__visualizer.playSubjectVideoWithHeadGaze(mappingFunc, streamer)
        return

    def getPostDataFromSubjectVideo(self, subjectVideoPath, frameCount, tName = ''):
        if tName != '': tName = ' for ' + tName
        postData = np.zeros((frameCount, 162))
        i = 0
        for subjFrame in self.openVideo(subjectVideoPath):
            annotations = self.__estimator.estimateInputValuesWithAnnotations(subjFrame)
            #annotations = self._mappingFunc.calculateOutputValuesWithAnnotations(subjFrame)
            pose, pPoints, landmarks = annotations
            postLine = np.concatenate((pose, landmarks.reshape((landmarks.size,)),
                                       pPoints.reshape((pPoints.size,))), 0)
            print('\rGenerating PostData%s (%.2f)...' % (tName, 
                                                         i/frameCount*100), end = '\r')   
            postData[i] = postLine
            i += 1
        return postData
    
    def _getPostDataAsGenerators(self, postData):
        inputValues = (l[:6] for l in postData)
        landmarks = (l[6:142].reshape((68, 2)) for l in postData)
        pPoints = (l[142:].reshape((10, 2)) for l in postData)
        return (inputValues, landmarks, pPoints)

    def replaySubjectVideoWithPostData(self, postData, subjectVideoPath):
        streamer = self.openVideo(subjectVideoPath)
        postDataGenerators = self._getPostDataAsGenerators(postData)
        self.__visualizer.replaySubjectVideoWithPostData(postDataGenerators, streamer)
        return
