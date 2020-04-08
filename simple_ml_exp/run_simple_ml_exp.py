from matplotlib import pyplot as plt
from .Analyzer import Analyzer
import math 
def testSimpleML(DataHandler):
    handler = DataHandler(readAllDataNow = False) 
    analyzer = Analyzer()
    data, postData = handler.getHeadPoseToPointingDataFor(3, 'random4')
    analyzer.testKalmanFilter(data, postData)
    #analyzer.print2DCorrelation(data, postData)(1, 'infinity')
    #print(data.shape, data2.shape)
    #pairs = handler.getAllHeadPoseToPointingPairs(2)
    #print()
    #print(pairs['random4'][1][:, 2].min())
    #for r in pairs['random4'][1][:20, :]:
    #    tv, rv = r[:3], r[3:]
    #    tv = str([t for t in tv])
    #    rv = str([math.degrees(t) for t in rv])
    #    #print('\r%s %s %s' % (s, tv, rv), end = '\r')
    #    print('%s %s' % (tv, rv))     
    #for i in range(1, 4):
    #    print(i)
    #    pairs = handler.getAllHeadPoseToPointingPairs(i)
    #    analyzer.printMinMaxForSet(pairs)

def main(DataHandler):
   testSimpleML(DataHandler)

if __name__ == '__main__':
    raise NotImplementedError