from datetime import datetime
from . import Paths
import cv2
import os

class VideoMerger(object):
        
    def initializeRecorder(self, id, trailName, fps = 30, dims = (1920, 1080)):
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        dir = Paths.MergedVideosFolder + ('%s%s' % (id, Paths.sep))
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok = True)
        now = str(datetime.now())[:-7].replace(':', '-').replace(' ', '_')
        recordName = trailName + '_%s_%s_merged.avi' % (id, now)
        return  cv2.VideoWriter(dir + recordName, fourcc, fps, dims)

    def merge(self, id, trailName, subjectVideoPath, trailVideoPath):
        subj = cv2.VideoCapture(subjectVideoPath)
        trail = cv2.VideoCapture(trailVideoPath)
        
        recorder = self.initializeRecorder(id, trailName)
        frameCount = 0
        while(True):
            subjRet, subjFrame = subj.read()
            subjFrame = cv2.flip(subjFrame, 1)
            trailRet, trailFrame = trail.read()
            if not (subjRet and trailRet):
                if frameCount < 1:
                    print('Something Wrong')
                break
            merged = cv2.addWeighted(subjFrame, 0.4, trailFrame, 1, 0)
            recorder.write(merged)
            #print('\rMerging frames (%d)...' % frameCount, end = '\r')
            frameCount += 1
        subj.release()       
        trail.release()
        recorder = None
        print('\r%s and %s have been merged.' % \
            (subjectVideoPath.split(Paths.sep)[-1], trailName), end = '\r')
        return

    def mergeSubjectVideoWithTrail(self, id, subjectVideoPath):
        subjectVideoName = subjectVideoPath.split(Paths.sep)[-1].split('_')
        trail = '_'.join(subjectVideoName[:subjectVideoName.index(id)])
        trailVideoPath = Paths.TrailVideosFolder + trail + '.avi'
        self.merge(id, trail, subjectVideoPath, trailVideoPath)

    def mergeAllSubjectVideos(self, id):
        subjectVideoFolder = Paths.SubjectsFolder + id + Paths.sep 
        subjectVideoList = os.listdir(subjectVideoFolder)
        for subjectVideoName in subjectVideoList:
            subjectVideoPath = subjectVideoFolder + subjectVideoName
            self.mergeSubjectVideoWithTrail(id, subjectVideoPath)