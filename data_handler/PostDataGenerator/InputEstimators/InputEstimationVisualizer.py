from PostDataGenerator.InputEstimators.MappingFunctions import Boundary, StaticMapping, DynamicMapping
import cv2

class InputEstimationVisualizer(object):
    
    def addBox(self, frame, pPoints):
        color = (255, 255, 255)
        cv2.polylines(frame, [pPoints], True, color, 2, cv2.LINE_AA)
        if len(pPoints) > 4:
            _pPoints = []
            for start, end in [(1,6), (2, 7), (3, 8)]:
                p = (tuple(pPoints[start]), tuple(pPoints[end]))
                _pPoints.append(p)
            for start, end in _pPoints:
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

    def addAllInputs(self, frame, pPoints = None,
                     landmarks = None, outputValues = None):    
        if not landmarks is None:
            frame = self.addLandmarks(frame, landmarks.astype(int))        
        if not pPoints is None:
            frame = self.addBox(frame, pPoints.astype(int))
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

    def showFrameWithAllInputs(self, frame, pPoints = None,
                     landmarks = None, outputValues = None, delay = 1):
        frame = self.addAllInputs(frame, pPoints, landmarks, outputValues)
        return self.showFrame(frame, delay)
    
    def playSubjectVideoWithAllInputs(self, estimator, streamer):
        for frame in streamer:
            annotations = estimator.estimateInputValuesWithAnnotations(frame)
            inputValues, pPoints, landmarks = annotations
            k = self.showFrameWithAllInputs(frame, pPoints, landmarks)
            if not k:
                break
        return
    
    def playSubjectVideoWithHeadGaze(self, mappingFunc, streamer):
        for frame in streamer:
            annotations = mappingFunc.calculateOutputValuesWithAnnotations(frame)
            outputValues, inputValues, pPoints, landmarks = annotations
            pp = mappingFunc.getEstimator().poseCalculator.calculate3DScreen()
            frame = self.addBox(frame, pp.astype(int))
            k = self.showFrameWithAllInputs(frame, pPoints, landmarks, inputValues)
            if not k:
                break
        return
    
    def replaySubjectVideoWithPostData(self, postData, streamer):
        jointStreamer = zip(*(postData + (streamer,)))
        for inputValues, landmarks, pPoints, frame in jointStreamer:
            print(pPoints.shape)
            k = self.showFrameWithAllInputs(frame, pPoints, landmarks)
            if not k:
                break
        return