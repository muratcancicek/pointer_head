from PostDataGenerator.InputEstimators.MappingFunctions import Boundary, StaticMapping, DynamicMapping
from PostDataGenerator.InputEstimators.InputEstimationVisualizer import InputEstimationVisualizer
from Paths import CV2Res10SSD_frozen_face_model_path
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import io

class Scene3DVisualizer(InputEstimationVisualizer):
    
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
        print(model_points)
        return model_points
    
    # define a function which returns an image as numpy array from figure, pad_inches=0'tight', bbox_inches=bbox
    def get_img_from_fig(self, fig, dpi=125.88):
        buf = io.BytesIO()
        #bbox = fig.bbox_inches.from_bounds(2.3, 1.3, 7.7, 4.3)
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def plot3DPoints(self, points = None, dpi=125.88):
        if points is None:
            points = self.get_full_model_points(CV2Res10SSD_frozen_face_model_path)
        s = 1
        fig = pyplot.figure(num=None, figsize=(s * 17.6, s * 14), dpi=dpi)
        ax = Axes3D(fig)
        ax.view_init(elev=-90, azim=90) 
        #pyplot.show()
        #buf = self.fig2data(fig)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-220, 220)
        ax.set_ylim(-50, 300)
        ax.set_zlim(-50, 800)
        ax.plot(points[-10:, 0], points[-10:, 1], points[-10:, 2])
        ax.plot(points[:10, 0], points[:10, 1], points[:10, 2])
        ax.invert_yaxis()
        buf = self.get_img_from_fig(fig, dpi=dpi)
        pyplot.close()
        return buf

    def playSubjectVideoWithHeadGaze(self, mappingFunc, streamer):
        i = -1
        for frame in streamer:
            annotations = mappingFunc.calculateOutputValuesWithAnnotations(frame)
            outputValues, inputValues, pPoints, landmarks = annotations
            landmarks3d = mappingFunc.getEstimator().poseCalculator.calculate3DLandmarks()
            screen = mappingFunc.getEstimator().poseCalculator.get3DScreen()
            nose = mappingFunc.getEstimator().poseCalculator.get3DNose()
            all3DPoints = np.concatenate((screen, landmarks3d, nose))
            f = self.plot3DPoints(all3DPoints)
            h, w, _ = f.shape
            xb, yb = int(w/5), int(h/5)
            s = 1.5
            xe, ye = xb + 3*xb, yb + 3*yb
            f = f[yb:ye, xb:xe]
            print(f.shape)
            k = self.showFrame(f)
            #frame = self.addBox(frame, pp.astype(int))
            #k = self.showFrameWithAllInputs(frame, pPoints, landmarks, inputValues)
            if not k:
                break
            #break
        return
    