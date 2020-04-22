from .DataHandler import DataHandler
from .VideoMerger import VideoMerger

def testDataHandler():
    handler = DataHandler(readAllDataNow = True) 
    #handler.getHeadPoseToPointingDataFor(1, 'infinity')
    #subjectTrail = handler.saveAllPostDataForSubject(2)'zigzag'
    #handler.playSubjectTrailWithHeadGaze(2, 'zigzag')
    #handler.playSubjectTrailWithAllInputs(2, 'random4')#(2, 'vertical')(1, 'infinity')
    #handler.play3DSubjectTrailWithHeadGaze(3, 'random4')(3, 'random4')
    #handler.savePostDataFromSubjectVideo(1, 'infinity')
    for i in range(5, 6): handler.saveAllPostDataForSubject(i)
    #handler.replaySubjectVideoWithPostData(3, 'random4')#(2, 'vertical')
    #handler.record3DSubjectTrailWithHeadGaze(1, 'zigzag')
    #handler.record3DSubjectTrailWithHeadGaze(2, 'zigzag')
    #for i in range(1, 4):
    #    handler.record3DSubjectTrailWithHeadGaze(i, 'infinity')(2, 'vertical')#
    #handler.replay3DSubjectTrailWithHeadGaze(3, 'random4')
    
def testVideoMerger():
    #VideoMerger().mergeAllSubjectVideos('1')
    handler = DataHandler(readAllDataNow = False) 
    id = 2
    subjectTrail = handler.readSubjectTrail(id, 'zigzag')
    VideoMerger().mergeSubjectVideoWithTrail(str(id), 
                                             subjectTrail['VideoPath'])

def main():
   testDataHandler()
    # testVideoMerger()
   #testFace()

if __name__ == '__main__':
    raise NotImplementedError
