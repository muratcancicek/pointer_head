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
        
    def plotAx(self, ax1, data, ind, yLabel, yLim = None): 
        colors = ['g', 'r', 'y', 'b', 'c', 'm', 'y']
        types = ['-', '-', '--', '-', '-.', ':', '-.', ':']
        ax1.set_ylabel(yLabel)
        if yLim:
            ax1.set_ylim(*yLim)
        #ax1.set_xlim(200, 300)
        lines = []
        for i, (d, _) in enumerate(data):
            l, = ax1.plot(d[:, ind], ls=types[i], c=colors[i])
            lines.append(l)
        mse = lambda d: self.root_mean_squared_error(data[0][0][ind], d) 
        data = [data[0]] + [(d, l+' (RMSE: %.3f)' % mse(d)) for d,l in data[1:]]
        ax1.legend(lines, [l for d, l in data])


    def plotHeadGazeAndPointingFo(self, *data, title = 'Plot', 
                                  plot = True, yLim = True): 
        fig = plt.figure(dpi = 120, figsize = (21, 9)) # (42, 18)) # 
        if data[0][0].shape[-1] == 1:
            ax1 = fig.add_subplot(111)
        else:
            ax1 = fig.add_subplot(211)
        ax1.set_title(title)
        self.plotAx(ax1, data, 0, 'X', (0,1920) if yLim else None)
        #self.plotAx(ax1, data, 0, 'X', (-960, 1920+960) if yLim else None)
        if data[0][0].shape[-1] == 1:
            return fig

        ax2 = fig.add_subplot(212)
        self.plotAx(ax2, data, 1, 'Y', (0,1080) if yLim else None)
        #self.plotAx(ax2, data, 1, 'Y', (-540, 1080+540) if yLim else None)
        if plot:
            plt.show()
        return fig
    
    def getKalmanFiltered(self, data):
        #mf = [0] * data.shape[-1]
        #cf = [0.1] * data.shape[-1]
        #kf = []
        newData = np.zeros_like(data)
        #for m, c in zip(mf, cf):
        #    kf.append(KalmanFilter(initial_state_mean=m,
        #                           initial_state_covariance=c))
        #for i in range(len(data)):
        #    for e in range(len(data[i])):
        #        mf[e], cf[e] = kf[e].filter_update(mf[e], cf[e], data[i][e])
        #        newData[i][e] = mf[e]
        #kf kf= 
        newData[:, 0] = KalmanFilter().em(data[:, 0]).smooth(data[:, 0])[0][:, 0]
        newData[:, 1] = KalmanFilter().em(data[:, 1]).smooth(data[:, 1])[0][:, 0]
        return newData

    def saveHeadGazeFilterPlotssAsPDF(self, pairs, tName, path, plot = False):
        f = self.plotHeadGazeAndPointingFo(*pairs, yLim = True, plot = plot,
                                           title = 'HeadGazeFilters for '+tName)
        img = self.get_img_from_fig(f)
        mergedImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pdf = Image.fromarray(mergedImg)
        pdf.save(path)
    
    def getPairsOfHeadGazeFiltersFor(self, handlers, subjId, tName):
        last = handlers[-2][1].getHeadGazeToPointingDataFor(subjId, tName) 
        target, last = (last[0], 'Target'), (last[1], handlers[-2][0])
        pairs = [(h.getHeadGazeToPointingDataFor(subjId, tName)[1], fltr)
                 for fltr, h in handlers[:-2]+[handlers[-1]]]
        pairs = pairs[:-1] + [last, pairs[-1]]
        kFiltered = (self.getKalmanFiltered(last[0]), 'FilteredGaze')
        pairs = [target] + pairs + [kFiltered]
        #pairs = [(d[:100], f) for d, f in pairs]
        return pairs
    
    def plotHeadGazeFiltersFor(self, handlers, subjId, tName, path):
        pairs = self.getPairsOfHeadGazeFiltersFor(handlers, subjId, tName)
        #f = self.plotHeadGazeAndPointingFo(*pairs, yLim = True, plot = True,
        #                                   title = 'HeadGazeFilters for '+tName)
        self.saveHeadGazeFilterPlotssAsPDF(pairs, tName, path, plot = False)

    def plotHeadGazeFiltersForSubj(self, handlers, subjId, Paths):
        trails = handlers[0][1].readAllTrails()
        for tName in trails:
            name = '%s_%s_HeadGazeFilters.pdf' % (tName, subjId)
            path = Paths.HeadGazeGraphsFolder + subjId + Paths.sep + name
            self.plotHeadGazeFiltersFor(handlers, subjId, tName, path)


    def mean_squared_error(self, y, y_hat): 
        return np.square(y - y_hat).mean()

    def root_mean_squared_error(self, y, y_hat): 
        return np.sqrt(self.mean_squared_error(y, y_hat))

    def printRMSE(self, y, y_hat): 
        mse = self.root_mean_squared_error(y, y_hat)
        print('RMSE: %.3f' % mse)
        return mse

    def getHeadGazeFilterRMSEsFor(self, handlers, subjId, tName): 
        pairs = self.getPairsOfHeadGazeFiltersFor(handlers, subjId, tName)
        mse = lambda y, y_hat: self.root_mean_squared_error(y, y_hat)
        target = pairs[0][0]
        pairs = [(mse(target[:, 0], d[:, 0]), mse(target[:, 1], d[:, 1]), f)
                for d, f in pairs[1:]]
        titles, values = ['trails'], []
        for x, y, f in pairs:
            titles.extend((f+'_X', f+'_Y', f+'_Mean'))
            values.extend((x, y, (x + y)/2))
        titles = ','.join(titles)
        values = ','.join([tName] + ['%.2f' % v for v in values])
        return titles, values

    def getHeadGazeFilterRMSEsForSubj(self, handlers, subjId): 
        trails = handlers[0][1].readAllTrails()
        e = lambda tName:self.getHeadGazeFilterRMSEsFor(handlers, subjId, tName)
        rmses = [e(tName) for tName in trails]
        titles = rmses[0][0]
        values = '\n'.join([v for t, v in rmses])
        return titles, values

    def getHeadGazeFilterCorrelFor(self, target, gaze, tName): 
        xCorr, yCorr = self.get2DCorrelation(target, gaze)
        titles = ','.join(['trails', 'X_Corr', 'Y_Corr', 'Mean_Corr'])
        values = ['%.2f' % v for v in (xCorr, yCorr, (xCorr + yCorr)/2)]
        values = ','.join([tName] + values)
        return titles, values

    def getHeadGazeFilterCorrelForSubj(self, pairs): 
        corrs = [self.getHeadGazeFilterCorrelFor(target, gaze, tName)
                for tName, (target, gaze) in pairs.items()]
        titles = corrs[0][0]
        values = '\n'.join([v for t, v in corrs])
        return titles, values

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
        title += ' (rmse=%.3f)' % self.root_mean_squared_error(y, y_hat)
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