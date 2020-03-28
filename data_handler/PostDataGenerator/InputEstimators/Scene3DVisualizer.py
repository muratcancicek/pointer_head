from PostDataGenerator.InputEstimators.MappingFunctions import Boundary, StaticMapping, DynamicMapping
from PostDataGenerator.InputEstimators.InputEstimationVisualizer import InputEstimationVisualizer
from Paths import CV2Res10SSD_frozen_face_model_path
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

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

    def plot3DPoints(self, points = None):
        if points is None:
            points = self.get_full_model_points(CV2Res10SSD_frozen_face_model_path)
        fig = pyplot.figure()
        ax = Axes3D(fig)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.plot(points[-10:, 0], points[-10:, 1], points[-10:, 2])
        ax.plot(points[:10, 0], points[:10, 1], points[:10, 2])
        ax.invert_yaxis()
        ax.view_init(elev=130, azim=-90) 
        pyplot.show()

    def playSubjectVideoWithHeadGaze(self, mappingFunc, streamer):
        i = -1
        for frame in streamer:
            i += 1 
            if i % 30 != 0:
                continue

            annotations = mappingFunc.calculateOutputValuesWithAnnotations(frame)
            outputValues, inputValues, pPoints, landmarks = annotations
            landmarks3d = mappingFunc.getEstimator().poseCalculator.calculate3DLandmarks()
            screen = mappingFunc.getEstimator().poseCalculator.get3DScreen()
            nose = mappingFunc.getEstimator().poseCalculator.get3DNose()
            all3DPoints = np.concatenate((screen, landmarks3d, nose))
            self.plot3DPoints(all3DPoints)
            print(nose)
            #frame = self.addBox(frame, pp.astype(int))
            #k = self.showFrameWithAllInputs(frame, pPoints, landmarks, inputValues)
            #if not k:
            #    break
            #break
        return
    