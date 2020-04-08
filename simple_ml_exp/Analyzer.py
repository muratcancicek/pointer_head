from matplotlib import pyplot as plt
from pykalman import KalmanFilter
import numpy as np
import math 

class Analyzer(object):
    def __init__(self):
        super()
        
    def scatterTwoSets(self, data, data2): 
        plt.scatter(data[:, 0], data[:, 1])
        plt.scatter(data2[:, 0], data2[:, 1])
        plt.show()

    def plotTwoSets(self, *dataSets): 
        for data in dataSets:
            plt.plot(data)
        plt.show()

    def getCorrelation(self, data, data2): 
        r = np.corrcoef(data, data2)
        #print(r)
        #print(r[0, 1])
        #print(r[1, 0])
        return r[0, 1]
        
    def get2DCorrelation(self, data, data2): 
        xCorr = self.getCorrelation(data[:, 0], data2[:, 0])        
        yCorr = self.getCorrelation(data[:, 1], data2[:, 1])
        return xCorr, yCorr
        
    def print2DCorrelation(self, data, data2): 
        xCorr, yCorr = self.get2DCorrelation(data, data2)
        print('Correlation on X-Axis: %.3f\n'\
              'Correlation on Y-Axis: %.3f' % (xCorr, yCorr))
        
    def print2DCorrelationForSet(self, pairsSet): 
        print('tName X-Axis Y-Axis')
        for tName, (data, postData) in pairsSet.items():
            xCorr, yCorr = self.get2DCorrelation(data, postData)
            print('%s %.3f %.3f' % (tName, xCorr, yCorr))
            #self.print2DCorrelation(data, postData)
        
    def printMinMaxForSet(self, pairsSet): 
        print('tName X-Axis Y-Axis')
        for tName, (data, postData) in pairsSet.items():
            xCorr, yCorr = postData[:, 2].min(), postData[:, 2].max()
            print('%s %.3f %.3f' % (tName, xCorr, yCorr))
             
    def testKalmanFilter(self, data, data2): 
        w, h = 1920, 1080
        mns =[0, 0, 700, 0, 0, 0]# np.array([], dtype = float)
        cs = 6*[[192, 107, 100, math.pi, math.pi, math.pi]]#np.array(, dtype = float)
        #print(cs)
        #print(mns.shape, cs.shape)
        kf = KalmanFilter(initial_state_mean=mns, initial_state_covariance=cs, n_dim_state=6)

        #m, v = kf.filter(data2)
        #print(m)
        m = np.zeros_like(data2)
        mf, cf = [[0], [0], [700], [0], [0], [0]], 6*[[192, 107, 100, math.pi/1800, math.pi/1800, math.pi/1800]]
        for i, y in enumerate(data2):
            mf, cf = kf.filter_update(mf, cf, y)
            m[i] = mf[0]
            
        print(m.shape)
        data = data[:, 0]
        data2 = [math.degrees(r) for r in data2[:, 4]]
        m = [math.degrees(r) for r in m[:, 4]]

        self.plotTwoSets(data, data2, m)