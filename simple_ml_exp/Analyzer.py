from matplotlib import pyplot as plt
from pykalman import KalmanFilter
from PIL import Image 
import numpy as np
import math 
import cv2
import os
import io

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
             
    def getCov(self, data, data2): 
        return [np.cov(data2[:, i]) for i in range(6)]
             
    def printCovForSet(self, pairsSet): 
        print('tName X-Axis Y-Axis')
        for tName, (data, postData) in pairsSet.items():
            cov = self.getCov(data, postData)
            print('%s %.3f %.3f %.3f %.3f %.3f %.3f' % (tName, *cov))
            
    def plotHeadGazeAndPointingFor(self, pointing, gaze): 
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.set_title('HeadGazeAndPointing')
        ax1.set_ylim(-960, 1920+960)
        ax1.set_ylabel('X')
        ax1.plot(pointing[:, 0])
        ax1.plot(gaze[:, 0])

        ax2 = fig.add_subplot(212)
        ax2.set_ylim(-540, 1080+540)
        ax2.set_ylabel('Y')
        ax2.plot(pointing[:, 1])
        ax2.plot(gaze[:, 1])
        plt.show()

    def plotHeadGazeAndPointingFo(self, *data, title = 'Plot', 
                                  plot = True, yLim = True): 
        fig = plt.figure(figsize = (21, 9))
        #fig = plt.figure(figsize = (63, 27), dpi = 200)
        ax1 = fig.add_subplot(211)
        ax1.set_title(title)
        if yLim:
            ax1.set_ylim(-960, 1920+960)
        ax1.set_ylabel('X')
        #ax1.set_xlim(200, 300)
        lines = []
        for d, _ in data:
            l, = ax1.plot(d[:, 0])
            lines.append(l)
        ax1.legend(lines, [l for d, l in data])

        ax2 = fig.add_subplot(212)
        if yLim:
            ax2.set_ylim(-540, 1080+540)
        #ax2.set_xlim(200, 300)
        ax2.set_ylabel('Y')
        lines = []
        for d, _ in data:
            l, = ax2.plot(d[:, 1])
            lines.append(l)
        ax2.legend(lines, [l for d, l in data])
        if plot:
            plt.show()
        return fig

    def plotHead(self, subjId, tName): 
        p = 'C:\\cStorage\\Datasets\\WhiteBallExp\\PostData'
        i = 1
        t = 'infinity' # 'zigzag' # 
        fileName = '\\%d\\%s_PostData.csv'% (subjId, tName)
        paths = [
            (p + '_pnpRansac_kf' + fileName, 'PnP_RASNAC_KF'),
            (p + '_pnpRansac' + fileName, 'PnP_RASNAC'),
            (p + '_pnp' + fileName, 'PnP'),
            (p + '_pnp_kf' + fileName, 'PnP_KF'),
                 ]
        sets = [(np.loadtxt(p, delimiter=',')[:, :2], l) for p, l in paths]
        self.plotHeadGazeAndPointingFo(*sets, yLim = False)
        
    def mean_squared_error(self, y, y_hat): 
        return np.square(y - y_hat).mean()

    def printMSE(self, y, y_hat): 
        mse = np.square(y - y_hat).mean() 
        print('MSE: %.3f' % mse)
        return mse
    
    def get_img_from_fig(self, fig, dpi=125.88):
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        buf = cv2.imdecode(img_arr, 1)
        return buf
    
    def plotPrediction(self, y, y_hat, y_gd = None, title = '', plot = True): 
        if title == '': title = 'PoseAndPointing'
        g = [y, y_hat]
        t = ['y', 'y_hat']
        if not y_gd is None:
            g = [y_gd] + g
            t = ['y_gd'] + t
        plots = zip(g, t)
        title += ' (mse=%.3f)' % self.mean_squared_error(y, y_hat)
        f = self.plotHeadGazeAndPointingFo(*plots, title = title, plot = plot)
        #plt.close()
        return f

    def getPredictionPlot(self, y, y_hat, y_gd = None, title = '', plot=False): 
        f = self.plotPrediction(y, y_hat, y_gd, title, plot)
        img = self.get_img_from_fig(f)
        return img

    def getPredictionResultImages(self, results, path, text, plot = False):
        figures = []
        for y_r, y_hat_r, y_gd, title in results:
            f = self.getPredictionPlot(y_r, y_hat_r, y_gd, title, plot)
            figures.append(f)
        return figures

    def getModelSummaryImage(self, sampleImg, text):
        shape = (50 * (len(text) + 2), ) + sampleImg.shape[1:]
        img = np.ones(shape, dtype = sampleImg.dtype)*255
        font = cv2.FONT_HERSHEY_SIMPLEX
        p = (int(img.shape[1]/4), 50)
        for l in text:
            img = cv2.putText(img, l, p, font, 1, (0, 0, 0), 1, cv2.LINE_AA)
            p = p[0], p[1] + 50
        return img

    def getPredResultImagesMerged(self, results, path, text, plot=False):
        images = self.getPredictionResultImages(results, path, text, plot)
        mergedImg = images[0]
        sumImg = self.getModelSummaryImage(mergedImg, text)
        images.append(sumImg)
        for f in images[1:]:
            mergedImg = np.concatenate((mergedImg, f))
        return mergedImg

    def savePredictionResults(self, results, path, text, plot = False):
        mergedImg = self.getPredResultImagesMerged(results, path, text, plot)
        cv2.imwrite(path, mergedImg)

    def savePredictionResultsAsPDF(self, results, path, text, plot = False):
        mergedImg = self.getPredResultImagesMerged(results, path, text, plot)
        mergedImg = cv2.cvtColor(mergedImg, cv2.COLOR_BGR2RGB)
        pdf = Image.fromarray(mergedImg)
        pdf.save(path)