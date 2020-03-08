import os, numpy as np

class DataHandler(object):
    def __init__(self, dataFolder = 'data/', readAllDataNow = False):
        super()
        self.dataFolder = dataFolder
        self.trails = {}
        if readAllDataNow:
            self.readAllTrails()

    def readTrailMetaData(self, file):
        keys = file.readline()[:-1].split(',')
        print(keys)
        values = file.readline()[:-1].split(',')
        tName = values[0]
        self.trails[tName] = {}
        self.trails[tName]['meta'] = {k: int(v) for k, v in zip(keys, values)
                                      if v.isnumeric()}
        self.trails[tName]['meta']['frameCount'] = float(values[3])
        self.trails[tName]['meta']['name'] = tName
        return tName, self.trails[tName]
    
    def readTrailSummary(self, tName, file):
        file.readline()
        keys = file.readline()[:-1].split(',')
        summary = {k: [] for k in keys}
        for l in range(self.trails[tName]['meta']['cornerCount']):
            values = file.readline()[:-1].split(',')
            if len(values) < len(keys):
                values.append(0)
            for i in range(len(keys)):
                summary[keys[i]].append(int(values[i]) if i != 0 else values[i])
        self.trails[tName]['summary'] = summary
        return self.trails[tName]['summary']
    
    def readTrailGroundTruth(self, tName, file):
        file.readline()
        self.trails[tName]['data'] = \
            np.zeros((self.trails[tName]['meta']['frameCount'], 2))
        for l in range(self.trails[tName]['meta']['frameCount']):
            values = file.readline()[:-1].split(',')
            self.trails[tName]['data'][l, :] = [int(v) for v in values]
        return self.trails[tName]['data']
    
    def readTrail(self, fileName):
        file = open(self.dataFolder + fileName, 'r')
        tName, _ = self.readTrailMetaData(file)
        self.readTrailSummary(tName, file)
        self.readTrailGroundTruth(tName, file)
        file.close()
        print('\r%s has been read' % tName, end = '\r')
        return self.trails[tName]

    def readAllTrails(self):
        trailList = os.listdir(self.dataFolder)
        for fileName in trailList:
            self.readTrail(fileName)
        return self.trails
    

