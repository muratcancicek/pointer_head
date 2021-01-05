from .DataHandler import DataHandler
from .VideoMerger import VideoMerger

def testDataHandler():
    handler = DataHandler(readAllDataNow = True) 
    #handler.getHeadPoseToPointingDataFor(1, 'infinity')
    #subjectTrail = handler.saveAllPostDataForSubject(2)'zigzag'
    #handler.playSubjectTrailWithHeadGaze(2, 'zigzag')
    #handler.replay3DSubjectTrailWithHeadGaze(3, 'random5')
    handler.replay3DSubjectTrailWithHeadGaze(3, 'zigzag_part1_slow')
    #handler.savePostDataFromSubjectVideo(3, 'zigzag_part1_slow')#(1, 'infinity')(2, 'infinity')
    #for i in range(5, 6): handler.saveAllPostDataForSubject(i)'vertical'
    #handler.playSubjectTrailWithAllInputs(3, 'random1')#
    #handler.replaySubjectVideoWithPostData(3, 'zigzag_part1_slow')#(2, 'vertical')
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
   #testVideoMerger()
   #testFace()

if __name__ == '__main__':
    raise NotImplementedError
