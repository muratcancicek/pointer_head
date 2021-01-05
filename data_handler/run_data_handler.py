from .DataHandler import DataHandler
from .VideoMerger import VideoMerger

def testDataHandler():
    handler = DataHandler(readAllDataNow = True) 
    #handler.getHeadPoseToPointingDataFor(1, 'infinity')
    #subjectTrail = handler.saveAllPostDataForSubject(2)'zigzag'
    #handler.replay3DSubjectTrailWithHeadGaze(2, 'infinity')
    #handler.savePostDataFromSubjectVideo(3, 'zigzag_part1_slow')
    #for i in range(5, 6): handler.saveAllPostDataForSubject(i)
    #handler.playSubjectTrailWithAllInputs(2, 'infinity')
    #handler.playSubjectVideoWithHeadGaze(2, 'infinity')
    handler.play3DSubjectTrailWithHeadGaze(2, 'infinity')
    #handler.replaySubjectVideoWithPostData(2, 'infinity')
    #handler.record3DSubjectTrailWithHeadGaze(3, 'infinity')

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
   #testVideoMerger()
   #testFace()

if __name__ == '__main__':
    raise NotImplementedError
