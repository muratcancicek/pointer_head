from .InputEstimators.MappingFunctions import Boundary, StaticMapping, DynamicMapping
from .InputEstimators.InputEstimationVisualizer import InputEstimationVisualizer
from .InputEstimators.Scene3DVisualizer import Scene3DVisualizer
from .InputEstimators.PoseEstimators import PoseEstimator, HeadGazer
#from .InputEstimators.LandmarkDetectors import LandmarkDetector
#from .InputEstimators.FaceDetectors import CVFaceDetector
from datetime import datetime
from .. import Paths
import numpy as np
import cv2
import os

class PostDataGenerator(object):
    
    gaze_b_ind = 0
    gaze_e_ind = 2
    pose_b_ind = gaze_e_ind
    pose_e_ind = gaze_e_ind + 6
    landmark_b_ind = pose_e_ind
    landmark_e_ind = landmark_b_ind + 136
    proPts_b_ind = landmark_e_ind
    proPts_e_ind = proPts_b_ind + 20

    def __init__(self):
        super()
        self.__estimator = HeadGazer() # PoseEstimator() # 
        #LandmarkDetector() # CVFaceDetector()
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
            annotations = \
                self.__estimator.estimateInputValuesWithAnnotations(subjFrame)
            inputValues, pPoints, landmarks = annotations
            k = self.__visualizer.showFrameWithAllInputs(subjFrame, 
                                                         pPoints, landmarks)
            recorder.write(subjFrame)
            print('\rMerging frames (%d)...' % frameCount, end = '\r')   
        recorder = None
        print('\r%s and %s have been merged.' % \
            (subjectVideoPath.split(Paths.sep)[-1], trailName), end = '\r')
        
    def playSubjectVideoWithAllInputs(self, subjectVideoPath):
        streamer = self.openVideo(subjectVideoPath)
        self.__visualizer.playSubjectVideoWithAllInputs(self.__estimator, 
                                                        streamer)
        return

    def _getMappingFunc(self, outputSize = (1920, 1080)):
        boundary = Boundary(0, outputSize[0], 0, outputSize[1])
        #self._mappingFunc = DynamicMapping(self.__estimator, boundary)
        return StaticMapping(self.__estimator, boundary)

    def playSubjectVideoWithHeadGaze(self, subjectVideoPath):
        mappingFunc = self._getMappingFunc()
        streamer = self.openVideo(subjectVideoPath)
        self.__visualizer.playSubjectVideoWithHeadGaze(mappingFunc, streamer)
        return
    
    def _getTrailStreamer(self, subjectVideoPath, id):
        subjectVideoName = subjectVideoPath.split(Paths.sep)[-1].split('_')
        trail = '_'.join(subjectVideoName[:subjectVideoName.index(id)])
        trailVideoPath = Paths.TrailVideosFolder + trail + '.avi'
        return self.openVideo(trailVideoPath)

    def play3DSubjectTrailWithHeadGaze(self, subjectVideoPath, id):
        trailStreamer = self._getTrailStreamer(subjectVideoPath, id)
        streamer = self.openVideo(subjectVideoPath)
        mappingFunc = self._getMappingFunc()
        self.__visualizer.playSubjectVideoWithHeadGaze(self.__estimator,
                                                      streamer, trailStreamer)
            
    def record3DSubjectTrailWithHeadGaze(self, subjectVideoPath, id):
        trailStreamer = self._getTrailStreamer(subjectVideoPath, id)
        streamer = self.openVideo(subjectVideoPath)
        mappingFunc = self._getMappingFunc()
        self.__visualizer.recordSubjectSceneVideoWithHeadGaze(mappingFunc, id,
                                                 trail, streamer, trailStreamer)

    def getPostDataFromSubjectVideo(self, subjectVideoPath, 
                                    frameCount, tName = ''):
        if tName != '': tName = ' for ' + tName
        postData = np.zeros((frameCount, 164))
        i = 0
        mappingFunc = self._getMappingFunc()
        for subjFrame in self.openVideo(subjectVideoPath):
            annotations = \
                self.__estimator.estimateInputValuesWithAnnotations(subjFrame)
            gaze, pPoints, landmarks = annotations
            pose = self.__estimator.getHeadPose()
            postLine = np.concatenate((gaze, pose, 
                                       landmarks.reshape((landmarks.size,)),
                                       pPoints.reshape((pPoints.size,))), 0)
            print('\rGenerating PostData%s (%.2f)...' % (tName, 
                                                         i/frameCount*100), 
                  end = '\r')   
            postData[i] = postLine
            i += 1
        return postData
    
    def _getPostDataAsGenerators(self, postData):
        PDG = PostDataGenerator
        headGazes = (l[PDG.gaze_b_ind:PDG.gaze_e_ind] for l in postData)
        poses = (l[PDG.pose_b_ind:PDG.pose_e_ind] for l in postData)
        landmarks = (l[PDG.landmark_b_ind:PDG.landmark_e_ind].reshape((68, 2)) 
                     for l in postData)
        pPoints = (l[PDG.proPts_b_ind:PDG.proPts_e_ind].reshape((10, 2)) 
                   for l in postData)
        return (headGazes, poses, landmarks, pPoints)

    def replaySubjectVideoWithPostData(self, postData, subjectVideoPath):
        streamer = self.openVideo(subjectVideoPath)
        postDataGenerators = self._getPostDataAsGenerators(postData)
        self.__visualizer.replaySubjectVideoWithPostData(postDataGenerators, 
                                                         streamer)
        return

    def replay3DSubjectTrailWithPostData(self, postData, subjectVideoPath, id):
        trailStreamer = self._getTrailStreamer(subjectVideoPath, id)
        streamer = self.openVideo(subjectVideoPath)
        print(postData.shape)
        postDataGenerators = self._getPostDataAsGenerators(postData)
        self.__visualizer.replay3DSubjectTrailWithPostData(postDataGenerators, 
                                                    streamer, self.__estimator,
                                                    trailStreamer)
        return
