from PostDataGenerator.PostDataGenerator import PostDataGenerator
import os, numpy as np
import Paths

class DataHandler(object):
    def __init__(self, dataFolder = Paths.TrailsDataFolder, readAllDataNow = False):
        super()
        self.dataFolder = dataFolder
        self.trails = {}
        self.subjects = {}
        if readAllDataNow:
            self.readAllTrails()

    def readTrailMetaData(self, file):
        keys = file.readline()[:-1].split(',')
        values = file.readline()[:-1].split(',')
        tName = values[0]
        self.trails[tName] = {}
        self.trails[tName]['meta'] = {k: int(v) for k, v in zip(keys, values)
                                      if v.isnumeric()}
        self.trails[tName]['meta']['duration'] = float(values[1])
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
    
    def readTrail(self, tName):
        if '.csv' != tName[-4:]: 
                tName += '.csv'
        file = open(self.dataFolder + tName, 'r')
        tName, _ = self.readTrailMetaData(file)
        self.readTrailSummary(tName, file)
        self.readTrailGroundTruth(tName, file)
        file.close()
        print('\r%s has been read' % tName, end = '\r')
        return self.trails[tName]

    def getTrail(self, tName):
        if '.csv' == tName[-4:]: 
            tName = tName[-4:]
        if tName in self.trails:
            return self.trails[tName]
        else:
            return self.readTrail(tName)
    
    def readAllTrails(self):
        trailList = os.listdir(self.dataFolder)
        for fileName in trailList:
            self.getTrail(fileName)
        return self.trails
        
    def addSubject(self, id):
        if isinstance(id, int): id = str(id)
        if id in self.subjects:
            return self.subjects[id]
        self.subjects[id] = {}
        self.subjects[id]['Folder'] = Paths.SubjectsFolder + id + Paths.sep
        self.subjects[id]['VideoList'] = os.listdir(self.subjects[id]['Folder'])
        return self.subjects[id]
    
    def __getSubjectVideoTrailName(self, id, subjectVideoName):
        subjectVideoNameParts = subjectVideoName.split('_')
        return '_'.join(subjectVideoNameParts[:subjectVideoNameParts.index(id)])

    def __findSubjectVideoName(self, id, tName):
        subjectVideoNames = [s for s in self.subjects[id]['VideoList']
                            if tName == self.__getSubjectVideoTrailName(id, s)]
        return subjectVideoNames[-1]

    def _readSubjectTrail(self, id, tName):
        self.trails[tName] = self.getTrail(tName)
        subjectVideoName = self.__findSubjectVideoName(id, tName)
        self.subjects[id][tName] = {}
        self.subjects[id][tName]['t'] = self.trails[tName]
        subjectVideoPath = self.subjects[id]['Folder'] + subjectVideoName
        self.subjects[id][tName]['VideoPath'] = subjectVideoPath
        return self.subjects[id][tName]

    def readSubjectTrail(self, id, tName):
        if isinstance(id, int): id = str(id)
        self.addSubject(id)
        return self._readSubjectTrail(id, tName)
    
    def readAllSubjectTrails(self, id):
        if isinstance(id, int): id = str(id)
        self.addSubject(id)
        for subjectVideoName in self.subjects[id]['VideoList']:
            tName = self.__getSubjectVideoTrailName(id, subjectVideoName)
            self._readSubjectTrail(id, tName)
        return  self.subjects[id]
    
    def __playSubjectTrailWith(self, id, tName):
        if isinstance(id, int): id = str(id)
        self.readSubjectTrail(id, tName)
        gen = PostDataGenerator()
        return gen, self.subjects[id][tName]['VideoPath']

    def playSubjectTrailWithAllInputs(self, id, tName):
        gen, path = self.__playSubjectTrailWith(id, tName)
        gen.playSubjectVideoWithAllInputs(path)

    def playSubjectTrailWithHeadGaze(self, id, tName):
        gen, path = self.__playSubjectTrailWith(id, tName)
        gen.playSubjectVideoWithHeadGaze(path)

    def generatePostDataFromSubjectVideo(self, id, tName):
        if isinstance(id, int): id = str(id)
        self.readSubjectTrail(id, tName)
        postDataGenerator = PostDataGenerator()
        path = self.subjects[id][tName]['VideoPath']
        c =  self.subjects[id][tName]['t']['meta']['frameCount']
        return postDataGenerator.getPostDataFromSubjectVideo(path, c, tName)

    def savePostDataFromSubjectVideo(self, id, tName):
        if isinstance(id, int): id = str(id)
        postData = self.generatePostDataFromSubjectVideo(id, tName)
        fileName = self.subjects[id][tName]['t']['meta']['name']+'_PostData.csv'
        path = Paths.PostDataFolder + id + Paths.sep + fileName 
        np.savetxt(path, postData, delimiter=',')
        print('\r%s has been saved successfully.' % fileName, end = '\r')  

    def saveAllPostDataForSubject(self, id):
        if isinstance(id, int): id = str(id)
        self.readAllSubjectTrails(id)
        for subjectVideoName in self.subjects[id]['VideoList']:
            tName = self.__getSubjectVideoTrailName(id, subjectVideoName)
            self.savePostDataFromSubjectVideo(id, tName) 

    def loadPostDataOfSubjectVideo(self, id, tName):
        if isinstance(id, int): id = str(id)
        self.readSubjectTrail(id, tName)
        fileName = self.subjects[id][tName]['t']['meta']['name']+'_PostData.csv'
        path = Paths.PostDataFolder + id + Paths.sep + fileName 
        return np.loadtxt(path, delimiter=',')
    
    def replaySubjectVideoWithPostData(self, id, tName):
        if isinstance(id, int): id = str(id)
        postData = self.loadPostDataOfSubjectVideo(id, tName)
        postDataGenerator = PostDataGenerator()
        path = self.subjects[id][tName]['VideoPath']
        postDataGenerator.replaySubjectVideoWithPostData(postData, path)