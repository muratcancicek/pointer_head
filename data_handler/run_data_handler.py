from DataHandler import DataHandler
from VideoMerger import VideoMerger

def testDataHandler():
    handler = DataHandler(readAllDataNow = False) 
    #subjectTrail = handler.saveAllPostDataForSubject(2)'zigzag'
    #handler.record3DSubjectTrailWithHeadGaze(3, 'zigzag')
    handler.play3DSubjectTrailWithHeadGaze(1, 'infinity')
    #handler.playSubjectTrailWithHeadGaze(2, 'zigzag')
    #handler.playSubjectTrailWithAllInputs(3, 'zigzag')
    #handler.savePostDataFromSubjectVideo(1, 'infinity')
    #for i in range(1, 4):
    #    handler.saveAllPostDataForSubject(i)
    #handler.replaySubjectVideoWithPostData(1, 'infinity')
    #handler.replay3DSubjectTrailWithHeadGaze(1, 'infinity')
    
def testVideoMerger():
    #VideoMerger().mergeAllSubjectVideos('1')
    handler = DataHandler(readAllDataNow = False) 
    id = 2
    subjectTrail = handler.readSubjectTrail(id, 'zigzag')
    VideoMerger().mergeSubjectVideoWithTrail(str(id),  subjectTrail['VideoPath'])

def main():
   testDataHandler()
    # testVideoMerger()
   #testFace()

if __name__ == '__main__':
    main()
