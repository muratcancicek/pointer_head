# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from .InputEstimatorABC import InputEstimatorABC
from .FaceDetectors import CVFaceDetector
from .LandmarkDetectors import LandmarkDetector
from ...Paths import CV2Res10SSD_frozen_face_model_path
from abc import ABC, abstractmethod
import numpy as np, math
from pykalman import KalmanFilter
import cv2

class HeadPoseEstimatorABC(InputEstimatorABC):

    def __init__(self, faceDetector = None, landmarkDetector = None, poseCalculator = None, *args, **kwargs):
        self._faceDetector = faceDetector
        self._landmarkDetector = landmarkDetector
        self._poseCalculator = poseCalculator
        self._headPose3D = np.zeros((3,))
        
    @abstractmethod
    def calculateHeadPose(self, frame):
        raise NotImplementedError
    
    @abstractmethod
    def _calculateHeadPoseWithAnnotations(self, frame):
        raise NotImplementedError
    
    def estimateInputValues(self, frame):
        return self.calculateHeadPose(frame)
    
    def estimateInputValuesWithAnnotations(self, frame):
        return self._calculateHeadPoseWithAnnotations(frame)

    @property
    def inputValues(self):
        return self._headPose3D

    def returns3D(self):
        return True

# The code is derived from the following repository:
# https://github.com/yinguobing/head-pose-estimation

class PoseCalculatorABC(ABC):
    @staticmethod
    def _get_points(rear_w, rear_h, rear_depth, front_w, front_h, front_depth):
        point_3d = []
        point_3d.append((-rear_w, -rear_h, rear_depth))
        point_3d.append((-rear_w, rear_h, rear_depth))
        point_3d.append((rear_w, rear_h, rear_depth))
        point_3d.append((rear_w, -rear_h, rear_depth))
        point_3d.append((-rear_w, -rear_h, rear_depth))
                
        point_3d.append((-front_w, -front_h, front_depth))
        point_3d.append((-front_w, front_h, front_depth))
        point_3d.append((front_w, front_h, front_depth))
        point_3d.append((front_w, -front_h, front_depth))
        point_3d.append((-front_w, -front_h, front_depth))
        
        point_3d = np.array(point_3d, dtype='float32').reshape(-1, 3)
        return point_3d

    @staticmethod
    def _get_3d_points(rear_size = 7.5, rear_depth = 0, front_size = 10.0, front_depth = 10.0):
        return PoseCalculatorABC._get_points(-rear_size, -rear_size, rear_depth, -front_size, -front_size, front_depth)

    def __init__(self, *args, **kwargs):
        self._pose = np.zeros((3,))
        self._rectCorners3D = self._get_3d_points()
        self._front_depth = 100
        self._rectCorners3D = self._get_3d_points(rear_size = 50, rear_depth = 0, 
                                                  front_size = 50, front_depth = self._front_depth)
        self._projectionPoints = None
        super().__init__(*args, **kwargs)
        
    @abstractmethod
    def calculatePose(self, shape):
        raise NotImplementedError

    def calculateProjectionPoints(self, shape, recalculatePose = False):
        if recalculatePose:
                self.calculatePose(shape)
        if not (self._rotation_vector is None or self._translation_vector is None):
            point_2d, _ = cv2.projectPoints(self._rectCorners3D, 
                                          self._rotation_vector, self._translation_vector, 
                                          self._camera_matrix, self._dist_coeffs)
            self._projectionPoints = np.int32(point_2d.reshape(-1, 2))
        return self._projectionPoints

    @property
    def pose(self):
        return self._pose
    
class YinsKalmanFilteredHeadPoseCalculator(PoseCalculatorABC):
   
    @staticmethod
    def _getCameraMatrix(size):
        scale = 3840/size[0]
        focal_length = [2667.497359647048143, 2667.497359647048143]
        focal_length = [l/scale for l in focal_length]
        camera_center = (1991.766193951624246, 1046.480313913574491)
        camera_center = [l/scale for l in camera_center]
        camera_matrix = np.array(
            [[focal_length[0], 0, camera_center[0]],
             [0, focal_length[1], camera_center[1]],
             [0, 0, 1]], dtype="double")
        return camera_matrix

    @staticmethod
    def __get_full_model_points(filename):
        """Get all 68 3D model points from file"""
        raw_values = []
        with open(filename) as file:
            for line in file:
                raw_values.append(line)
        model_points = np.array(raw_values, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        return model_points

    def __get_kalman_filter(self):
        w, h = 1920, 1080
        self._mf = [0, 0, 0, 0, 0, 0]
        self._cf = [0.001, 0.001, 0.001, math.pi/1800, math.pi/1800, math.pi/1800]
        #self._cf = 6*[[0.001, 0.001, 0.001, math.pi/1800, math.pi/1800, math.pi/1800]]
        self._kf = []
        for m, c in zip(self._mf, self._cf):
            self._kf.append(KalmanFilter(initial_state_mean=m, initial_state_covariance=c))
        #self._mf = [[m] for m in self._mf]
        return self._kf

    
    def __init__(self, face_model_path = None, inputFramesize = (1920, 1080), *args, **kwargs):
        super().__init__(*args, **kwargs)
        if face_model_path == None:
            face_model_path = CV2Res10SSD_frozen_face_model_path
        self._faceModelPoints = self.__get_full_model_points(face_model_path)
        self._inputFramesize = inputFramesize 
        self._front_depth = 100
        self._rectCorners3D = self._get_3d_points(rear_size = 80, rear_depth = 0, 
                                                  front_size = 10, front_depth = self._front_depth)
        # Camera internals
        self._camera_matrix = self._getCameraMatrix(inputFramesize)
        self._dist_coeffs = np.array([[0.2562583722261407293], [-0.5884400171468063823], 
                                      [0.001658348839202715592], [-0.0006434617243149612104]
                                      ,[0.3660073010818283845]])
        self._rotation_vector = np.array([[-0.0], [0.0], [-0.0]]) 
        self._translation_vector = np.array([[0.0], [0.0], [550.0]])
        self._kf = self.__get_kalman_filter()

    def solve_pose_by_68_points(self, image_points): 
        image_points = image_points.astype('float32')
        (_, rotation_vector, translation_vector) = \
            cv2.solvePnP(self._faceModelPoints, image_points,
                        self._camera_matrix, self._dist_coeffs,
                         rvec=self._rotation_vector, 
                         tvec=self._translation_vector, useExtrinsicGuess=True)
        self._rotation_vector = rotation_vector
        self._translation_vector = translation_vector
        return (rotation_vector, translation_vector)

    def calculatePose(self, shape):
        pose = self.solve_pose_by_68_points(shape)
        self._pose = np.concatenate((self._translation_vector, 
                                     self._rotation_vector), 0)
        for i, kf in enumerate(self._kf):
            self._mf[i], self._cf[i] = self._kf[i].filter_update(self._mf[i], 
                                                                 self._cf[i],
                                                                 self._pose[i])
        self._pose[:] = self._mf
        self._pose[0] *= -1  # X-axis is reverse for the model
        self._pose[3] *= -1
        self._rotation_vector = self._pose[3:]
        self._translation_vector = self._pose[:3]
        return self._pose

class PoseEstimator(HeadPoseEstimatorABC):
    def __init__(self, faceDetector = None, landmarkDetector = None, 
                 poseCalculator = None, face_landmark_path = None, 
                 inputFramesize = (1920, 1080), *args, **kwargs):
        if landmarkDetector == None:
            if faceDetector == None:
                faceDetector = CVFaceDetector(squaringFaceBox = True)
            landmarkDetector = LandmarkDetector(faceDetector)
        if poseCalculator == None:
            poseCalculator = YinsKalmanFilteredHeadPoseCalculator(inputFramesize = inputFramesize)
        self._headPose3D = np.zeros((3,))
        super().__init__(faceDetector, landmarkDetector, poseCalculator, *args, **kwargs)
    
    def calculateHeadPose(self, frame):
        self._landmarks = self._landmarkDetector.detectFacialLandmarks(frame)
        if len(self._landmarks) == 0:
            return self._headPose3D
        else:
            self._headPose3D = self._poseCalculator.calculatePose(self._landmarks)
            return self._headPose3D
            
    def _calculateHeadPoseWithAnnotations(self, frame):
        self._headPose3D = self.calculateHeadPose(frame)
        self._pPoints = self._poseCalculator.calculateProjectionPoints(self._landmarks)
        return self._headPose3D, self._pPoints, self._landmarks

    @property
    def headPose(self):
        return self._headPose3D

    @property
    def poseCalculator(self):
        return self._poseCalculator

class MuratcansHeadGazeCalculator(YinsKalmanFilteredHeadPoseCalculator):
       
    def __init__(self, face_model_path = None, inputFramesize = (1920, 1080), *args, **kwargs):
        super().__init__(face_model_path, inputFramesize, *args, **kwargs)
        self._front_depth = 700
        self._rectCorners3D = self._get_3d_points(rear_size = 40, rear_depth = 0, 
                                                  front_size = 40, 
                                                  front_depth = self._front_depth)
        self._objectPointsVec = [self._faceModelPoints]
        self._imagePointsVec = []

    def calibrateCamera(self, imagePoints):
        ip = imagePoints.astype('float32')        
        #print(imagePoints)
        self._imagePointsVec.append(ip)
        n = 7
        if len(self._imagePointsVec) < n+1:
            return
        self._imagePointsVec.pop(0)
        #print(ip.shape, self._faceModelPoints.shape, 
        #      len(self._objectPointsVec), len(self._imagePointsVec))
        flags=(cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.SOLVEPNP_ITERATIVE)
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = \
            cv2.calibrateCamera(self._objectPointsVec, self._imagePointsVec, 
                                (1920, 1080), self._camera_matrix, 
                                self._dist_coeffs,  flags=flags)
        self._camera_matrix, self._dist_coeffs = cameraMatrix, distCoeffs
        self._rotation_vector, self._translation_vector = rvecs[0], tvecs[0]

    def calculateProjectionPointsAsGaze(self, shape, recalculatePose = False):
        if recalculatePose:
                self.calculatePose(shape)
        if not (self._rotation_vector is None or self._translation_vector is None):
            self._front_depth = np.linalg.norm(self._translation_vector) 
            self._rectCorners3D = \
                self._get_3d_points(rear_size = 0, rear_depth = 55, front_size = 0, 
                                    front_depth = 35*self._front_depth)
            rv = self._rotation_vector.copy()
            rv[1] *= -1
            rv[0] -= 0.2
            tv = self._translation_vector.copy()
            tv[0] *= -1
            point_2d, _ = cv2.projectPoints(self._rectCorners3D, rv, 
                                            tv, self._camera_matrix,
                                           self._dist_coeffs)
            self._projectionPoints = np.int32(point_2d.reshape(-1, 2))
        return self._projectionPoints

    def calculateHeadGazeProjection(self):
        output = 5*(self.get3DNose()[-1] -
                    self.get3DScreen()[3])
        output[0] *= -1
        return output[:-1]

    def calculateHeadGazeWithProjectionPoints(self, shape):
        self._pose = self.calculatePose(shape)
        self._projectionPoints = self.calculateProjectionPointsAsGaze(shape)
        output = self.calculateHeadGazeProjection()
        return output, self._projectionPoints

    def calculateReverseHeadGazeWithProjectionPoints(self, shape):
        pose = self.solve_pose_by_68_points(shape)
        output = self.calculateHeadGazeProjection()
        output[0] = self._inputFramesize[0]-output[0]
        self._rotation_vector[0, 0] *= -1
        self._rotation_vector[1, 0] *= -1
        #self._translation_vector[0, 0] *= -1
        rv =  np.array([math.degrees(t[0]) for t in self._rotation_vector])
        self._pose = np.concatenate((self._translation_vector[:,0], rv), 0)
        self._projectionPoints = self.calculateProjectionPointsAsGaze(shape)
        return output, self._projectionPoints

    def translateTo3D(self, points):
        rot = self._rotation_vector.copy()
        rot[0] *= -1
        rotation_mat, _ = cv2.Rodrigues(rot)
        t_vec = self._translation_vector.copy()
        t_vec[0] *= -1
        project_mat = cv2.hconcat((rotation_mat, t_vec))
        project_mat = np.concatenate((project_mat, np.zeros((1, 4))), 0)
        project_mat[-1, -1] = 1
        points_ = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
        points3d = np.matmul(project_mat, points_.T).T[:, :-1]
        return points3d
    
    def updatePose(self, pose):
        self._pose = pose
        self._translation_vector = pose[:3].reshape((3, 1))
        self._rotation_vector = np.array([t for t in pose[3:]])
        self._rotation_vector = self._rotation_vector.reshape((3, 1))
        self._front_depth = self._translation_vector[2, 0] 
    
    def get3DNose(self):
        nose =  self._get_3d_points(rear_size = 0, 
                                    rear_depth = 0, front_size = 0, 
                                    front_depth = self._front_depth)
        nose = self.translateTo3D(nose)
        p1, p2 = nose[0], nose[-1]; dist = p1 - p2
        norm = np.linalg.norm(dist); unit = dist/norm
        nose[-int(nose.shape[0]/2):] = p2 - (p2[-1]/unit[-1]) * unit
        return nose

    def get3DScreen(self):
        t_vec = np.array([[0], [162], [0.0]])
        return t_vec.T + self._get_points(192, 107, 0, 192, 107, 0)

    def calculate3DScreen(self):
        translation_vector = np.array([[0], [0], [0.0]])
        rotation_vector = np.array([[-0.0], [0.0], [-0.0]])
        corners3D = self._get_points(192, 107, 700, 192, 107, 267)
        point_2d, _ = cv2.projectPoints(corners3D, rotation_vector, 
                                        translation_vector, self._camera_matrix,
                                        self._dist_coeffs)
        projectionPoints = np.int32(point_2d.reshape(-1, 2))
        return projectionPoints

    def calculate3DLandmarks(self):
        face = self._faceModelPoints.copy()
        return self.translateTo3D(face)
    
    def calculateAll3DPoints(self):
        landmarks3d = self.calculate3DLandmarks()
        screen = self.get3DScreen()
        nose = self.get3DNose()
        all3DPoints = np.concatenate((screen, landmarks3d, nose))
        return all3DPoints

    def calculate3DProjection(self, points):
        translation_vector = np.array([[0], [0], [0.0]])
        rotation_vector = np.array([[-0.0], [0.0], [-0.0]])
        point_2d, _ = cv2.projectPoints(points, rotation_vector, 
                                        translation_vector, self._camera_matrix,
                                        self._dist_coeffs)
        projectionPoints = np.int32(point_2d.reshape(-1, 2))
        return projectionPoints

    def calculate3DScreenProjection(self):
        screen = self.get3DScreen()
        return self.calculate3DProjection(screen)

    def calculate3DLandmarksProjection(self):
        landmarks3d = self.calculate3DLandmarks()
        return self.calculate3DProjection(landmarks3d)

    def calculate3DNoseProjection(self):
        nose = self.get3DNose()
        return self.calculate3DProjection(nose)

    def calculateAll3DProjections(self):
        screenProj = self.calculate3DScreenProjection()
        landmarksProj = self.calculate3DLandmarksProjection()
        noseProj = self.calculate3DNoseProjection()
        return screenProj, landmarksProj, noseProj

class HeadGazer(PoseEstimator):
    def __init__(self, faceDetector = None, landmarkDetector = None, 
                 poseCalculator = None, face_landmark_path = None, 
                 inputFramesize = (1920, 1080), *args, **kwargs):
        if poseCalculator == None:
            poseCalculator = \
                MuratcansHeadGazeCalculator(inputFramesize = inputFramesize)
        self._pPoints = np.zeros((1, 2))
        self._gazingFrameSize = inputFramesize
        self._halfFrameHeight = inputFramesize[1]/2
        super().__init__(faceDetector, landmarkDetector, poseCalculator, 
                         face_landmark_path, inputFramesize, *args, **kwargs)
        
    def calculateHeadPose(self, frame):
        self._landmarks = self._landmarkDetector.detectFacialLandmarks(frame)
        if len(self._landmarks) == 0:
            return self._headPose3D
        else:
            self._headPose3D = \
                self._poseCalculator.calculatePose(self._landmarks)
            return self._headPose3D
                    
    def calculateHeadGaze(self, frame):
        self._landmarks = self._landmarkDetector.detectFacialLandmarks(frame)
        if len(self._landmarks) != 0:
            self._halfFrameHeight = frame.shape[0]/2
            self._headPose3D, self._pPoints = self._poseCalculator\
                .calculateHeadGazeWithProjectionPoints(self._landmarks) 
            return self._headPose3D
            
    def _calculateHeadPoseWithAnnotations(self, frame):
        self._headPose3D = self.calculateHeadGaze(frame)
        return self._headPose3D, self._pPoints, self._landmarks
            
    def calculateHeadPoseWithAnnotations(self, frame, landmarks = None):
        if landmarks is None:
            self._headPose3D = self.calculateHeadGaze(frame)
        else:
            self._landmarks = landmarks
            if len(self._landmarks) != 0:
                self._halfFrameHeight = frame.shape[0]/2
                g = self._poseCalculator\
                    .calculateHeadGazeWithProjectionPoints(self._landmarks) 
                self._headPose3D, self._pPoints = g
        return self._headPose3D, self._pPoints, self._landmarks

    def estimateReverseInputValuesWithAnnotations(self, frame):
        self._landmarks = self._landmarkDetector.detectFacialLandmarks(frame)
        if len(self._landmarks) != 0:
            self._halfFrameHeight = frame.shape[0]/2
            g = self._poseCalculator\
                .calculateReverseHeadGazeWithProjectionPoints(self._landmarks) 
            self._headPose3D, self._pPoints = g
        return self._headPose3D, self._pPoints, self._landmarks
    
    def getHeadPose(self):
        return  self._poseCalculator.pose

    def getGazingFrameDimensions(self):
        #return int(1920), int(self._halfFrameHeight + 1080)
        #print(self._gazingFrameSize)
        return self._gazingFrameSize
