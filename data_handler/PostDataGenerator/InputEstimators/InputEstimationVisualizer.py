from .MappingFunctions import Boundary, StaticMapping, DynamicMapping
import cv2

class InputEstimationVisualizer(object):
    
    def addBox(self, frame, pPts):
        color = (255, 255, 255)
        cv2.polylines(frame, [pPts], True, color, 2, cv2.LINE_AA)
        if len(pPts) > 4:
            _pPts = []
            for start, end in [(1,6), (2, 7), (3, 8)]:
                p = (tuple(pPts[start]), tuple(pPts[end]))
                _pPts.append(p)
            for start, end in _pPts:
                cv2.line(frame, start, end, color, 2, cv2.LINE_AA)
        return frame
    
    def addLandmarks(self, frame, landmarks, c = (255, 0, 0)):
        for i, (x, y) in enumerate(landmarks):
            #if not i in [39, 42]:
            #    continue
            cv2.circle(frame, (x, y), 6, c, -1, cv2.LINE_AA)
        return frame
     
    def addPointer(self, frame, outputValues):
        #boundaries = self._mappingFunc.getOutputBoundaries()
        outputSize = (1920, 1080)
        outputValues[0] = outputSize[0] - outputValues[0]
        boundaries = Boundary(0, outputSize[0], 0, outputSize[1])
        (height, width, depth) = frame.shape
        (xRange, yRange, _) = boundaries.getRanges()
        if xRange != width or yRange != height:
            xRange, yRange = boundaries.getVolumeAbsRatio(outputValues)
            x, y = int(xRange*width), int(yRange*height)
        else:
            x, y = outputValues[:2].astype(int)
        cv2.circle(frame, (x, y), 1, (0, 0, 235), 56, cv2.LINE_AA)
        return frame

    def addAllInputs(self, frame, pPts = None,
                     landmarks = None, outputValues = None):    
        if not landmarks is None:
            frame = self.addLandmarks(frame, landmarks.astype(int))        
        if not pPts is None:
            frame = self.addBox(frame, pPts.astype(int))
        if not outputValues is None:
            frame = self.addPointer(frame, outputValues.astype(int))
        return frame

    def showFrame(self, frame, delay = 1):
        cv2.imshow('frame', frame)
        k = cv2.waitKey(delay)
        if k == 27 or k == ord('q'):
            return False
        else:
            return True
    
    def __addText(self, frame, text, pos, color, largeScale = True):
        if largeScale:
            cv2.putText(frame, text, pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, thickness=8)
        else:
            cv2.putText(frame, text, pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness=1)
        return frame
    
    def _addValuesLineByLine(self, frame, values, labels,
                            position, colors, largeScale = True):
        for v, l, c in zip(values, labels, colors):
            text = "{:s}: {:7.2f}".format(l, float(v))
            frame = self.__addText(frame, text, position, c, largeScale)
            position = (position[0], position[1]+(70 if largeScale else 30))
        return frame

    def _addValues(self, inputValues, frame, 
                   pos = (20, 60), prefix = '', largeScale = True):
        labels = [prefix+l for l in ['X', 'Y', 'Z']]
        g = 0 if largeScale else 200
        colors = ((0, 0, 255), (0, 255, 0), (255, g, 0))
        return self._addValuesLineByLine(frame, inputValues, labels,
                                         pos, colors, largeScale)

    def _addMeasurements(self, inputValues, pose, frame, largeScale = True):
        initialPos, gap = ((20, 60), 200) if largeScale else ((120, 30), 90)
        frame = self._addValues(pose[:3], frame, pos = initialPos, 
                                prefix = 'Pos', largeScale = largeScale)
        initialPos = (initialPos[0], initialPos[1] + gap)
        frame = self._addValues(pose[3:], frame, pos =initialPos, 
                                prefix = 'Or', largeScale = largeScale)
        initialPos = (initialPos[0], initialPos[1] + gap)
        frame = self._addValues(inputValues, frame, pos = initialPos,
                                prefix = 'Gaze', largeScale = largeScale)
        return frame

    def showFrameWithAllInputs(self, frame, pPts = None, landmarks = None, 
                     outputValues = None, pose = None, delay = 1):
        frame = self.addAllInputs(frame, pPts, landmarks, outputValues)
        h, w, _ = frame.shape
        cv2.line(frame, (0, int(h/2)), (w, int(h/2)), (0,0,0), 5)
        cv2.line(frame, (int(w/2), 0), (int(w/2), h), (0,0,0), 5)
        if not pose is None:
            frame = self._addMeasurements(outputValues, pose, frame)
        return self.showFrame(frame, delay)

    def playSubjectVideoWithAllInputs(self, estimator, streamer):
        self._estimator = estimator
        for frame in streamer:
            annotations = \
                estimator.estimateInputValuesWithAnnotations(frame)
            pose = estimator.getHeadPose()
            inputValues, pPts, landmarks = annotations
            k = self.showFrameWithAllInputs(frame, pPts, 
                                            landmarks, inputValues, pose)
            if not k:
                break
        return
    
    def replaySubjectVideoWithPostData(self, postData, streamer):
        jointStreamer = zip(*(postData + (streamer,)))
        for headGaze, pose, landmarks, pPts, frame in jointStreamer:
            k = self.showFrameWithAllInputs(frame, pPts, 
                                            landmarks, headGaze, pose)
            if not k:
                break
        return