from .PostDataGenerator.PostDataGenerator import PostDataGenerator
import os, numpy as np
from . import Paths

class DataHandler(object):
    Paths = Paths
    def __init__(self, dataFolder = Paths.TrailsDataFolder, 
                 postDataFolder = Paths.PostDataFolder, readAllDataNow = False):
        super()
        self.dataFolder = dataFolder
        self.postDataFolder = postDataFolder
        self.analysisFolder = Paths.AnalysisFolder
        self.analysisCommonFolder = Paths.AnalysisCommonFolder
        self.Paths = Paths
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
                try: 
                    summary[keys[i]].append(int(values[i]) 
                                            if i != 0 else values[i])
                except ValueError:
                    summary[keys[i]].append(float(values[i]) 
                                            if i != 0 else values[i])

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
            tName = tName[:-4]
        if tName in self.trails:
            return self.trails[tName]
        else:
            return self.readTrail(tName)
    
    def readAllTrails(self):
        trailList = os.listdir(self.dataFolder)
        for fileName in trailList:
            if 'summaries' != fileName[:9]:
                self.getTrail(fileName)
        return self.trails
        
    def addSubject(self, id):
        if isinstance(id, int): id = str(id)
        if id in self.subjects:
            return self.subjects[id]
        self.subjects[id] = {}
        self.subjects[id]['ts'] = {}
        self.subjects[id]['pairs'] = {}
        self.subjects[id]['Folder'] = Paths.SubjectsFolder + id + Paths.sep
        self.subjects[id]['VideoList'] = []
        if os.path.isdir(self.subjects[id]['Folder']):
            self.subjects[id]['VideoList'] = \
                os.listdir(self.subjects[id]['Folder'])
        return self.subjects[id]
    
    def __getSubjectVideoTrailName(self, id, subjectVideoName):
        subjectVideoNameParts = subjectVideoName.split('_')
        return '_'.join(subjectVideoNameParts[:subjectVideoNameParts.index(id)])

    def __findSubjectVideoName(self, id, tName):
        subjectVideoNames = [s for s in self.subjects[id]['VideoList']
                            if tName == self.__getSubjectVideoTrailName(id, s)]
        if len(subjectVideoNames) > 0:
            return subjectVideoNames[-1]
        else:
            return 'NotExit'

    def _readSubjectTrail(self, id, tName):
        self.trails[tName] = self.getTrail(tName)
        self.subjects[id]['ts'][tName] = {}
        self.subjects[id]['ts'][tName]['t'] = self.trails[tName]
        subjectVideoName = self.__findSubjectVideoName(id, tName)
        subjectVideoPath = self.subjects[id]['Folder'] + subjectVideoName
        self.subjects[id]['ts'][tName]['VideoPath'] = subjectVideoPath
        return self.subjects[id]['ts'][tName]

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
        return gen, self.subjects[id]['ts'][tName]['VideoPath']

    def playSubjectTrailWithAllInputs(self, id, tName):
        gen, path = self.__playSubjectTrailWith(id, tName)
        gen.playSubjectVideoWithAllInputs(path)
        
    def playSubjectTrailWithHeadGaze(self, id, tName):
        gen, path = self.__playSubjectTrailWith(id, tName)
        gen.playSubjectVideoWithHeadGaze(path)

    def play3DSubjectTrailWithHeadGaze(self, id, tName):
        if isinstance(id, int): id = str(id)
        gen, path = self.__playSubjectTrailWith(id, tName)
        gen.play3DSubjectTrailWithHeadGaze(path, id)

    def record3DSubjectTrailWithHeadGaze(self, id, tName):
        if isinstance(id, int): id = str(id)
        gen, path = self.__playSubjectTrailWith(id, tName)
        gen.record3DSubjectTrailWithHeadGaze(path, id)

    def replaySubjectVideoWithPostData(self, id, tName):
        if isinstance(id, int): id = str(id)
        self.readSubjectTrail(id, tName)
        postDataGenerator = PostDataGenerator()
        path = self.subjects[id]['ts'][tName]['VideoPath']
        postData = self.loadPostDataOfSubjectVideo(id, tName)
        postDataGenerator.replaySubjectVideoWithPostData(postData, path)
    
    def replay3DSubjectTrailWithHeadGaze(self, id, tName):
        if isinstance(id, int): id = str(id)
        self.readSubjectTrail(id, tName)
        gen, path = self.__playSubjectTrailWith(id, tName)
        postData = self.loadPostDataOfSubjectVideo(id, tName)
        gen.replay3DSubjectTrailWithPostData(postData, path, id)
            
    def generatePostDataFromSubjectVideo(self, id, tName):
        if isinstance(id, int): id = str(id)
        self.readSubjectTrail(id, tName)
        postDataGenerator = PostDataGenerator()
        path = self.subjects[id]['ts'][tName]['VideoPath']
        c =  self.subjects[id]['ts'][tName]['t']['meta']['frameCount']
        return postDataGenerator.getPostDataFromSubjectVideo(path, c, tName)

    def savePostDataFromSubjectVideo(self, id, tName):
        if isinstance(id, int): id = str(id)
        postData = self.generatePostDataFromSubjectVideo(id, tName)
        f = self.subjects[id]['ts'][tName]['t']['meta']['name']+'_PostData.csv'
        path = self.postDataFolder + id + Paths.sep + f 
        np.savetxt(path, postData, delimiter=',')
        print('\r%s has been saved successfully.' % f, end = '\r')  

    def saveAllPostDataForSubject(self, id):
        if isinstance(id, int): id = str(id)
        self.readAllSubjectTrails(id)
        for subjectVideoName in self.subjects[id]['VideoList']:
            tName = self.__getSubjectVideoTrailName(id, subjectVideoName)
            self.savePostDataFromSubjectVideo(id, tName) 

    def loadPostDataOfSubjectVideo(self, id, tName):
        if isinstance(id, int): id = str(id)
        self.readSubjectTrail(id, tName)
        f = self.subjects[id]['ts'][tName]['t']['meta']['name']+'_PostData.csv'
        path = self.postDataFolder + id + Paths.sep + f 
        if os.path.isfile(path):
            self.subjects[id]['ts'][tName]['p'] = np.loadtxt(path, delimiter=',')
            return self.subjects[id]['ts'][tName]['p']
        else:
            print('\r%s does not exist, skipping...' % path, end = '\r')
            return None

    def getFakePostDataForSubject(self, id, fakeId, tName):
        if isinstance(id, int): id = str(id)
        postData = self.loadPostDataOfSubjectVideo(id, tName)
        if postData is None:
            return None
        postDataGenerator = PostDataGenerator()
        postData_fake = postDataGenerator.regenerateFakerPostData(postData)
        return postData_fake

    def saveFakePostDataForSubject(self, id, fakeId, tName):
        id = str(id); fakeId = str(fakeId)
        postData_fake = self.getFakePostDataForSubject(id, fakeId, tName)
        if postData_fake is None:
            return None
        f = self.subjects[id]['ts'][tName]['t']['meta']['name']+'_PostData.csv'
        path = self.postDataFolder + fakeId + Paths.sep
        os.makedirs(path, exist_ok = True)
        np.savetxt(path + f , postData_fake, delimiter=',')
        l = '\r%s (for Faker %s) has been saved successfully.' % (f, fakeId)
        print(l, end = '\r')  
        return

    def saveAllFakePostDataForSubject(self, id, fakeId):
        trails = self.readAllTrails()
        for tName in trails:
            self.saveFakePostDataForSubject(id, fakeId, tName)
        return

    def loadAllPostDataOfSubject(self, id):
        if isinstance(id, int): id = str(id)
        self.readAllTrails()
        self.addSubject(id)
        postDataSet = {}
        for tName in self.trails:
            self._readSubjectTrail(id, tName)
            p = self.loadPostDataOfSubjectVideo(id, tName)
            if not p is None:
                postDataSet[tName] = p
        return postDataSet
    
    def loadDatasetPairFor(self, subjId, tName):
        if isinstance(subjId, int): subjId = str(subjId)
        postData = self.loadPostDataOfSubjectVideo(subjId, tName)
        if not postData is None:
            self.subjects[subjId]['pairs'][tName] = \
                self.subjects[subjId]['ts'][tName]['t']['data'], postData
            return self.subjects[subjId]['pairs'][tName]
        else:
            return None
    
    def getDatasetPairFor(self, subjId, tName):
        if isinstance(subjId, int): subjId = str(subjId)
        self.addSubject(subjId)
        if tName in self.subjects[subjId]['pairs']:
            return self.subjects[subjId]['pairs'][tName]
        else:
            return self.loadDatasetPairFor(subjId, tName)
    
    def loadAllDatasetPairsFor(self, subjId):
        if isinstance(subjId, int): subjId = str(subjId)
        postDataSet = self.loadAllPostDataOfSubject(subjId)
        for tName in self.subjects[subjId]['ts']:
            if not tName in self.subjects[subjId]['pairs']:
                pair = self.getDatasetPairFor(subjId, tName)
                if pair:
                    self.subjects[subjId]['pairs'][tName] = pair                    
        return self.subjects[subjId]['pairs']
    
    def getAllDatasetPairsFor(self, subjId):
        if isinstance(subjId, int): subjId = str(subjId)
        return self.loadAllDatasetPairsFor(subjId)    
    
    def getHeadGazeToPointingDataFor(self, subjId, tName):
        if isinstance(subjId, int): subjId = str(subjId)
        pair = self.getDatasetPairFor(subjId, tName)
        if pair:
            data, postData = pair
            PDG = PostDataGenerator
            return data, postData[:, PDG.gaze_b_ind:PDG.gaze_e_ind]
            
    def getAllHeadGazeToPointingPairs(self, subjId): 
        if isinstance(subjId, int): subjId = str(subjId)
        pairsSet = self.getAllDatasetPairsFor(subjId)
        PDG = PostDataGenerator
        return {tName: (data, postData[:, PDG.gaze_b_ind:PDG.gaze_e_ind])
               for tName, (data, postData) in pairsSet.items()}
    
    def getHeadPoseToPointingDataFor(self, subjId, tName):
        if isinstance(subjId, int): subjId = str(subjId)
        pair = self.getDatasetPairFor(subjId, tName)
        if pair:
            data, postData = pair
            PDG = PostDataGenerator
            return data, postData[:, PDG.pose_b_ind:PDG.pose_e_ind]
            
    def getAllHeadPoseToPointingPairs(self, subjId): 
        if isinstance(subjId, int): subjId = str(subjId)
        pairsSet = self.getAllDatasetPairsFor(subjId)
        PDG = PostDataGenerator
        return {tName: (data, postData[:, PDG.pose_b_ind:PDG.pose_e_ind])
               for tName, (data, postData) in pairsSet.items()}
            
    def getAllHeadAngleToPointingPairs(self, subjId): 
        if isinstance(subjId, int): subjId = str(subjId)
        pairsSet = self.getAllDatasetPairsFor(subjId)
        PDG = PostDataGenerator
        return {tName: (data, postData[:, PDG.pose_b_ind+3:PDG.pose_e_ind])
               for tName, (data, postData) in pairsSet.items()}
            
    def getAllLandmarksToPointingPairs(self, subjId): 
        if isinstance(subjId, int): subjId = str(subjId)
        pairsSet = self.getAllDatasetPairsFor(subjId)
        PDG = PostDataGenerator
        return {tName: (data, postData[:, PDG.landmark_b_ind:PDG.landmark_e_ind])
               for tName, (data, postData) in pairsSet.items()}
            
    def getDefaultTestTrailsForSubj(self, subjId): 
        if isinstance(subjId, int): subjId = str(subjId)
        test = ['infinity_slow', 'random1', 'horizontal_part1_slow']
        train = [tName for tName in self.subjects[subjId]['ts']
                if not tName in test]
        return sorted(train), sorted(test)

    def getDefaultTestTrailsForSubjList(self, sList): 
        train0, test0 = self.getDefaultTestTrailsForSubj(sList[0])
        test, train = [], []
        for subjId in sList:
            for t in train0:
                train.append(t + '_' + subjId)
            for t in test0:
                test.append(t + '_' + subjId)
        return sorted(train), sorted(test)

    def regenerateGazeFromPose(self, pose):
        return PostDataGenerator().regenerateGazeFromPose(pose)