from .DataHandler import DataHandler
from .VideoMerger import VideoMerger

def testDataHandler():
    handler = DataHandler(readAllDataNow = True) 
    #handler.saveAllPostDataForSubject(2)
    #handler.savePostLandmarksFromSubjectVideo(5, 'infinity')
    #for i in range(1, 3): handler.saveAllPostDataForSubject(i)
    #for i in range(1, 9): handler.saveAllPostLandmarksForSubject(i)
    handler.playSubjectTrailWithLandmarks(2, 'infinity')
    #handler.playSubjectTrailWithAllInputs(2, 'infinity')
    #handler.playSubjectVideoWithHeadGaze(2, 'infinity')
    #handler.play3DSubjectTrailWithHeadGaze(5, 'infinity')
    #handler.replaySubjectVideoWithPostData(2, 'infinity')
    #handler.replay3DSubjectTrailWithHeadGaze(2, 'infinity')
    #handler.record3DSubjectTrailWithHeadGaze(1, 'zigzag')

    #for i in range(1, 4):
    #    handler.record3DSubjectTrailWithHeadGaze(i, 'infinity')(2, 'vertical')#
    #handler.replay3DSubjectTrailWithHeadGaze(3, 'random4')
    
    #handler.checkFrameCount(2)

def testVideoMerger():
    #VideoMerger().mergeAllSubjectVideos('1')
    handler = DataHandler(readAllDataNow = False) 
    id = 2
    subjectTrail = handler.readSubjectTrail(id, 'zigzag')
    VideoMerger().mergeSubjectVideoWithTrail(str(id), 
                                             subjectTrail['VideoPath'])

def main():
    testDataHandler()
   #testVideoMerger()
   #testFace()

if __name__ == '__main__':
    raise NotImplementedError
