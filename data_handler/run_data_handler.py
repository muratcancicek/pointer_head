from DataHandler import DataHandler
from VideoMerger import VideoMerger

def testDataHandler():
    handler = DataHandler(readAllDataNow = False)
    #trails = handler.readAllTrails()
    trail = handler.readTrail('random1.csv')
    print()
    print(trail['meta']['frameCount'])
    print(trail['data'].shape)
    print(trail['data'])
    print(trail['data'][-1])

def testVideoMerger():
    VideoMerger().mergeAllSubjectVideos('1')


def main():
    testDataHandler()
    #testVideoMerger()

if __name__ == '__main__':
    main()
