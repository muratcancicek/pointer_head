from .MappingFunctions import Boundary, StaticMapping, DynamicMapping
from .InputEstimationVisualizer import InputEstimationVisualizer
from ...Paths import CV2Res10SSD_frozen_face_model_path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from datetime import datetime
import numpy as np
from ... import Paths
import cv2
import os
import io

class Scene3DVisualizer(InputEstimationVisualizer):

    def hex_to_rgb(self, value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16)
                     for i in range(0, lv, lv // 3))

    def __init__(self, sceneScale = 1,
                landmarkColorStr = '#00ff00', screenColorStr = '#0000ff'):
        super()
        self._size = (1920, 1080)
        self._sceneScale = sceneScale
        self._landmarkColorStr = landmarkColorStr # (0, 255, 0)
        self._landmarkColor = tuple(reversed(self.hex_to_rgb(self._landmarkColorStr)))
        self._screenColorStr = screenColorStr # (0, 0, 255)
        self._screenColor = tuple(reversed(self.hex_to_rgb(self._screenColorStr)))
    
    def get_full_model_points(self, filename):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        # model_points *= 4
        model_points[:, 1] *= -1
        model_points[:, -1] *= -1
        return model_points
    
    # define a function which returns an image as numpy array from figure,
    #  pad_inches=0'tight', bbox_inches=bbox
    def get_img_from_fig(self, fig, dpi=125.88):
        buf = io.BytesIO()
        pyplot.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def configAx(self, ax):
        ax.view_init(elev=90, azim=-90) 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        s = self._sceneScale
        ax.set_xlim(-220*s, 220*s)
        ax.set_ylim(-50*s, 300*s)
        ax.set_zlim(-50*s, 800*s)
        ax.w_xaxis.set_pane_color((0., 0., 0., 0.))
        ax.w_yaxis.set_pane_color((0., 0., 0., 0.))
        ax.w_zaxis.set_pane_color((0., 0., 0., 0.))
        ax.grid(False)
        ax.invert_yaxis()
        ax.invert_zaxis()
        #ax.invert_xaxis()
        return ax
    
    def addPlane(self, ax, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #h, w, _ = img.shape
        #xb, yb = int(w/8), int(h/4)
        #xe, ye = 5*xb, yb + 2*yb
        #img = img[yb:ye, 3*xb:xe]
        img = cv2.resize(img, (int(383), int(214)))
        #s = 2
        #h, w, _ = img.shape
        #img = cv2.resize(img, (int(w/s), int(h/s)))
        img = img/255
        #img[:, :, :] = 1
        img = np.rot90(img,axes=(0,1))
        x, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        x = (x - x.shape[0]/2).astype(int)
        y = (y + 55).astype(int)
        z = 0*np.ones(x.shape)
        ax.plot_surface(x, y, z, rstride=1, cstride=1,
                        facecolors=img, alpha=.5, linewidth=0, zorder=-1)
        return ax

    def plot3DPoints(self, points = None, dpi=125.88, img = None, plot = False):
        if points is None:
            points = \
                self.get_full_model_points(CV2Res10SSD_frozen_face_model_path)
        pyplot.style.use('dark_background')
        s = 1
        fig = pyplot.figure(num=None, figsize=(s * 17.6, s * 14), dpi=dpi)
        ax = Axes3D(fig)
        ax = self.configAx(ax)
        if not img is None:
            ax = self.addPlane(ax, img)
        ax.scatter(points[:10, 0], points[:10, 1], points[:10, 2])
        ax.scatter(points[10:-11, 0], points[10:-11, 1],
                   points[10:-11, 2], c = self._landmarkColorStr)
        ax.scatter(points[-11:, 0], points[-11:, 1], points[-11:, 2])
        ax.plot(points[:10, 0], points[:10, 1], 
                points[:10, 2], c = self._screenColorStr)
        ax.plot(points[-10:, 0], points[-10:, 1], points[-10:, 2])
        if plot:
            pyplot.show()
        else:
            buf = self.get_img_from_fig(fig, dpi=dpi)
            pyplot.close()
            h, w, _ = buf.shape
            xb, yb = int(w/5), int(h/5)
            xe, ye = xb + 3*xb, yb + 3*yb
            buf = buf[yb:ye, xb:xe]
            return buf

    def getLargeFrame(self, frame, s = 2, xb = None, yb = None):
        largeFrame = np.zeros((s * frame.shape[0], s * frame.shape[1], 
                               frame.shape[2]), np.uint8)
        h, w, _ = largeFrame.shape
        if xb is None: xb = int(h/4)+167
        if yb is None: yb = 167
        largeFrame[yb:yb+frame.shape[0], xb:xb+frame.shape[1]] = frame
        #largeFrame = cv2.addWeighted(frame, 1, largeFrame, 1, 0)
        return largeFrame

    def getMergedLargeFrame(self, frame1, frame2, s = 2):
        largeFrame = self.getLargeFrame(frame1, s) # frame1 # 
        scene = np.zeros_like(largeFrame)
        h, w, _ = largeFrame.shape
        xb, yb = int((largeFrame.shape[1]-frame2.shape[1])/2)+20, 0 #int(h/5)
        scene[yb:yb+frame2.shape[0], xb:xb+frame2.shape[1]] = frame2
        merged = cv2.addWeighted(scene, 1, largeFrame, 1, 0)
        return merged
    
    def showScene(self, all3DPoints):
        self.plot3DPoints(all3DPoints, plot = True)
        return None
    
    def showSceneFrame(self, all3DPoints):
        scene = self.plot3DPoints(all3DPoints)
        return self.showFrame(scene)

    def find3DFaceInScene(self, frame, scene, landmarksProj):
        scene = (scene > 0).astype(np.uint8)*255
        points = np.where(np.all(scene == self._landmarkColor, axis=-1))
        points = np.array([[x, y] for y, x in set(zip(points[0], points[1]))])
        top_left = (points[:, 0].min(), points[:, 1].min())
        bottom_right = (points[:, 0].max(), points[:, 1].max())
        return top_left, bottom_right
    
    def addFaceToSceneFrame(self, frame, scene, landmarksProj):
        offset = 20
        xb, yb = landmarksProj[:, 0].min(), landmarksProj[:, 1].min()
        xe, ye = landmarksProj[:, 0].max(), landmarksProj[:, 1].max()
        temp = frame[yb-offset:ye+offset, xb-offset:xe+offset]
        top_left, bottom_right = \
           self.find3DFaceInScene(frame, scene, landmarksProj)
        #scene = cv2.rectangle(scene, top_left, bottom_right, (0,0,255), 4)
        w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
        offsetX_scaled = int(offset*w/temp.shape[1])
        offsetY_scaled = int(offset*h/temp.shape[0])
        w_scaled, h_scaled = w + 2*offsetX_scaled, h +  2*offsetY_scaled
        temp = cv2.resize(temp, (w_scaled,h_scaled))
        bg = np.zeros_like(scene)
        xb2, yb2 = top_left[0] - offsetX_scaled, top_left[1] - offsetY_scaled
        xb2, yb2 = max(xb2, 0), max(yb2, 0)
        bg[yb2:yb2+temp.shape[0], xb2:xb2+temp.shape[1]] = temp
        scene = cv2.addWeighted(bg, 1, scene, 1, 0)
        return scene
    
    def find3DScreenInScene(self, scene, trailFrame):
        scene = (scene > 0).astype(np.uint8)*255
        points = np.where(np.all(scene == self._screenColor, axis=-1))
        points = np.array([[x, y] for y, x in set(zip(points[0], points[1]))])
        top_left = (points[:, 0].min(), points[:, 1].min())
        bottom_right = (points[:, 0].max(), points[:, 1].max())
        return top_left, bottom_right

    def addTrailToSceneFrame(self, scene0, scene, trailFrame):
        top_left, bottom_right = self.find3DScreenInScene(scene0, trailFrame)
        w, h = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
        trailFrame = cv2.resize(trailFrame, (w, h))
        bg = np.zeros_like(scene)
        xb2, yb2 = top_left[0], top_left[1]
        xb2, yb2 = max(xb2, 0), max(yb2, 0)
        bg[yb2:yb2+trailFrame.shape[0], 
           xb2:xb2+trailFrame.shape[1]] = trailFrame
        scene = cv2.addWeighted(scene, 1, bg, 1, 0)
        return scene

    def addPointingValueToSceneFrame(self, scene, headGaze):
        font = cv2.FONT_HERSHEY_SIMPLEX
        s = '%d px, %d px' % tuple([int(i) for i in headGaze[:2]])
        p = (int(scene.shape[1]/2)-120,int(scene.shape[0]/2))
        scene = cv2.putText(scene, s, p, font, 1, (255,255,255), 1, cv2.LINE_AA)
        return scene
    
    def showSceneFrameWithFace(self, frame, all3DPoints,
                              estimator, trailStreamer = None):
        screenProj, landmarksProj, noseProj = \
           estimator.poseCalculator.calculateAll3DProjections()
        scene0 = self.plot3DPoints(all3DPoints)
        scene = self.addFaceToSceneFrame(frame, scene0, landmarksProj)
        if not trailStreamer is None:
            trailFrame = next(trailStreamer)
            scene = self.addTrailToSceneFrame(scene0, scene, trailFrame)
        headGaze = estimator.poseCalculator.calculateHeadGazeProjection()
        self.addPointingValueToSceneFrame(scene, headGaze)
        return scene, self.showFrame(scene)
    
    def showMergedLargeFrame(self, frame, all3DPoints, landmarks3d, landmarks):
        scene = self.plot3DPoints(all3DPoints)
        p, p2 = landmarks3d[39:40], landmarks3d[42:43]
        dist = np.linalg.norm(p[:, :-1]-p2[:, :-1])*129/50
        pp, pp2 = landmarks[39:40], landmarks[42:43]
        dist2 = np.linalg.norm(pp[:, :-1]-pp2[:, :-1])
        r = 1 if dist2 == 0 else dist/dist2
        #print(r)

        frame = self.addLandmarks(frame, landmarks.astype(int))
        frame = cv2.circle(frame, (int(frame.shape[1]/2),int(frame.shape[0]/2)),
                           15, (255, 55, 255), -1)
        h, w, _ = frame.shape
        xb, yb = int(w/8), int(h/4)
        xe, ye = 5*xb, yb + 2*yb
        frame = frame[yb:ye, 3*xb:xe]
        frame = cv2.resize(frame, (int(r*frame.shape[1]),int(r*frame.shape[0])))
        merged = self.getMergedLargeFrame(scene, frame, s = 2)
        merged = cv2.resize(merged, (int(merged.shape[1]/2),
                                    int(merged.shape[0]/2)))
        return self.showFrame(merged)
    
    def showSceneWithTrail(self, all3DPoints, trailStreamer):
        if not trailStreamer is None:
            trailFrame = next(trailStreamer) # , plot = True
            scene = self.plot3DPoints(all3DPoints, img = trailFrame)
            return self.showFrame(scene)
        
    def getPerspectiveTransformFrame(self, frame, landmarks, landmarksProj):
        warp_mat = cv2.getAffineTransform(landmarks[:3].astype(np.float32), 
                                          landmarksProj[:3].astype(np.float32))
        return cv2.warpAffine(frame, warp_mat, (frame.shape[1], frame.shape[0]))
        
    def showProjectedFrame(self, frame, mappingFunc, landmarks):
        screenProj, landmarksProj, noseProj = \
           mappingFunc.getEstimator().poseCalculator.calculateall3DProjections()
        frame = self.addLandmarks(frame, landmarks.astype(int))
        frame = self.addLandmarks(frame, landmarksProj.astype(int), (0, 0, 255))
        h, w, _ = frame.shape
        xb, yb = int(w/8), int(h/4)
        xe, ye = 5*xb, yb + 2*yb
        frame2 = frame[yb:ye, 3*xb:xe]
        frame = np.zeros_like(frame)
        frame[yb:ye, 3*xb:xe] = frame2 
        #elf.getPerspectiveTransformFrame(frame2, landmarks, landmarksProj)
        frame = self.getPerspectiveTransformFrame(frame, landmarks, 
                                                  landmarksProj)
        #return self.showFrame(frame)
        xb, yb = int((frame.shape[1])/2), 0 #int(h/5)
        #print(screenProj)
        largeFrame = self.getLargeFrame(frame, 2, xb, yb) # frame1 # 
        #noseProj[:, 0] += xb
        #frame = self.addBox(largeFrame, noseProj.astype(int))
        ##frame = self.addBox(frame, noseProj.astype(int))
        return self.showFrame(largeFrame)

    def playSubjectVideoWithHeadGaze(self, estimator, 
                                     streamer, trailStreamer = None):
        if not trailStreamer is None:
            trailStreamer = (cv2.flip(tf, 1) for tf in trailStreamer)
        for frame in streamer:
            estimator.estimateInputValuesWithAnnotations(frame)
            all3DPoints = estimator.poseCalculator.calculateAll3DPoints()
            #k = self.showScene(all3DPoints)
            #k = self.showSceneFrame(all3DPoints)
            #k = self.showSceneFrameWithFace(frame, all3DPoints, mappingFunc)
            f, k = self.showSceneFrameWithFace(frame, all3DPoints,
                                               estimator, trailStreamer)
            #k = self.showSceneWithTrail(all3DPoints, trailStreamer)
            #k = self.showProjectedFrame(frame, mappingFunc, landmarks)
            #k = self.showMergedLargeFrame(frame, all3DPoints,
            #                             landmarks3d, landmarks)
            if not k:
                break
        return
    
    def initializeRecorder(self, id, trailName, fps = 30, dims = (1920, 1080)):
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        dir = Paths.MergedVideosFolder + ('%s%s' % (id, Paths.sep))
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok = True)
        now = str(datetime.now())[:-7].replace(':', '-').replace(' ', '_')
        recordName = trailName + '_%s_%s_merged3DScene.avi' % (id, now)
        print(dir + recordName, 'will be written')
        return  cv2.VideoWriter(dir + recordName, fourcc, fps, dims)
 
    def _write(self, recorder, frame):
        background = np.zeros((self._size[1], self._size[0], 3))
        xb = int((self._size[0]-frame.shape[1])/2)
        yb = int((self._size[1]-frame.shape[0])/2)
        xe, ye = xb+frame.shape[1], yb+frame.shape[0]
        background[yb:ye, xb:xe] = frame
        recorder.write(background.astype(np.uint8))

    def recordSubjectSceneVideoWithHeadGaze(self, mappingFunc, id, trailName, 
                                     streamer, trailStreamer = None):
        recorder = self.initializeRecorder(id, trailName, dims = self._size)
        for frame in streamer:
            frame = cv2.flip(frame, 1)
            annotations=mappingFunc.calculateOutputValuesWithAnnotations(frame)
            outputValues, inputValues, pPoints, landmarks = annotations
            all3DPoints = \
                mappingFunc.getEstimator().poseCalculator.calculateAll3DPoints()
            frame, k = self.showSceneFrameWithFace(frame, all3DPoints,
                                                   mappingFunc, trailStreamer)
            self._write(recorder, frame)
            if not k:
                recorder.release()
                break
        recorder.release()
        return
    
    def replay3DSubjectTrailWithPostData(self, postData, streamer, 
                                         estimator, trailStreamer = None):
        jointStreamer = zip(*(postData + (streamer,)))
        if not trailStreamer is None:
            trailStreamer = (cv2.flip(tf, 1) for tf in trailStreamer)
        for headGaze, pose, landmarks, pPts, frame in jointStreamer:
            estimator.poseCalculator.updatePose(pose)
            all3DPoints = estimator.poseCalculator.calculateAll3DPoints()
            #k = self.showScene(all3DPoints)
            #k = self.showSceneFrame(all3DPoints)
            #k = self.showSceneFrameWithFace(frame, all3DPoints, estimator)
            f, k = self.showSceneFrameWithFace(frame, all3DPoints,
                                            estimator, trailStreamer)
            if not k:
                break
        return
    