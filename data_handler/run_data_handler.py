from DataHandler import DataHandler
from VideoMerger import VideoMerger

def testDataHandler():
    handler = DataHandler(readAllDataNow = False) 
    #subjectTrail = handler.saveAllPostDataForSubject(2)
    handler.playSubjectTrailWithAllInputs(1, 'infinity')
    #handler.savePostDataFromSubjectVideo(1, 'infinity')
    #handler.replaySubjectVideoWithPostData(1, 'zigzag')

def testVideoMerger():
    #VideoMerger().mergeAllSubjectVideos('1')
    handler = DataHandler(readAllDataNow = False) 
    id = 2
    subjectTrail = handler.readSubjectTrail(id, 'zigzag')
    VideoMerger().mergeSubjectVideoWithTrail(str(id),  subjectTrail['VideoPath'])

def main():
   testDataHandler()
    # testVideoMerger()

if __name__ == '__main__':
    main()
