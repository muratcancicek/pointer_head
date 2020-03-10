from DataHandler import DataHandler
from VideoMerger import VideoMerger

def testDataHandler():
    handler = DataHandler(readAllDataNow = False)
    #trails = handler.readAllTrails()
    subjectTrail = handler.readSubjectTrail(1, 'random1')
    print(subjectTrail)

def testVideoMerger():
    VideoMerger().mergeAllSubjectVideos('1')


def main():
    testDataHandler()
    #testVideoMerger()

if __name__ == '__main__':
    main()
