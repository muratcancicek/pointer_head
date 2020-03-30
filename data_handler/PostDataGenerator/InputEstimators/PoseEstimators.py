# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from PostDataGenerator.InputEstimators.InputEstimatorABC import InputEstimatorABC
from PostDataGenerator.InputEstimators.FaceDetectors import CVFaceDetector
from PostDataGenerator.InputEstimators.LandmarkDetectors import LandmarkDetector
from Paths import CV2Res10SSD_frozen_face_model_path
from abc import ABC, abstractmethod
import numpy as np, math
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
    def _get_3d_points(rear_size = 7.5, rear_depth = 0, front_size = 10.0, front_depth = 10.0):
        point_3d = []
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))
                
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        
        point_3d = np.array(point_3d, dtype='float32').reshape(-1, 3)
        return point_3d

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
   
    class Stabilizer:
        """Using Kalman filter as a point stabilizer."""

        def __init__(self, state_num=4, measure_num=2, cov_process=0.0001, cov_measure=0.1):
            """Initialization"""
            # Currently we only support scalar and point, so check user input first.
            assert state_num == 4 or state_num == 2, "Only scalar and point supported, Check state_num please."

            # Store the parameters.
            self.state_num = state_num
            self.measure_num = measure_num

            # The filter itself.
            self.filter = cv2.KalmanFilter(state_num, measure_num, 0)

            # Store the state.
            self.state = np.zeros((state_num, 1), dtype=np.float32)

            # Store the measurement result.
            self.measurement = np.array((measure_num, 1), np.float32)

            # Store the prediction.
            self.prediction = np.zeros((state_num, 1), np.float32)

            # Kalman parameters setup for scalar.
            if self.measure_num == 1:
                self.filter.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)

                self.filter.measurementMatrix = np.array([[1, 1]], np.float32)

                self.filter.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * cov_process

                self.filter.measurementNoiseCov = np.array( [[1]], np.float32) * cov_measure

            # Kalman parameters setup for point.
            if self.measure_num == 2:
                self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                         [0, 1, 0, 1],
                                                         [0, 0, 1, 0],
                                                         [0, 0, 0, 1]], np.float32)

                self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                          [0, 1, 0, 0]], np.float32)

                self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], np.float32) * cov_process

                self.filter.measurementNoiseCov = np.array([[1, 0],
                                                            [0, 1]], np.float32) * cov_measure

        def update(self, measurement):
            """Update the filter"""
            # Make kalman prediction
            self.prediction = self.filter.predict()

            # Get new measurement
            if self.measure_num == 1:
                self.measurement = np.array([[np.float32(measurement[0])]])
            else:
                self.measurement = np.array([[np.float32(measurement[0])],
                                             [np.float32(measurement[1])]])

            # Correct according to mesurement
            self.filter.correct(self.measurement)

            # Update state value.
            self.state = self.filter.statePost

        def set_q_r(self, cov_process=0.1, cov_measure=0.001):
            """Set new value for processNoiseCov and measurementNoiseCov."""
            if self.measure_num == 1:
                self.filter.processNoiseCov = np.array([[1, 0],
                                                        [0, 1]], np.float32) * cov_process
                self.filter.measurementNoiseCov = np.array(
                    [[1]], np.float32) * cov_measure
            else:
                self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], np.float32) * cov_process
                self.filter.measurementNoiseCov = np.array([[1, 0],
                                                            [0, 1]], np.float32) * cov_measure
   
    @staticmethod
    def _getCameraMatrix(size):
        scale = 3840/size[0]
        focal_length = [2667.497359647048143, 2667.497359647048143]
        focal_length = [l/scale for l in focal_length]
        #print(focal_length)
        #print()
        #print()
        #print()
        camera_center = (1991.766193951624246, 1046.480313913574491)
        camera_center = [l/scale for l in camera_center]
        camera_matrix = np.array(
            [[focal_length[0], 0, camera_center[0]],
             [0, focal_length[1], camera_center[1]],
             [0, 0, 1]], dtype="double")
        return camera_matrix

    @staticmethod
    def __get_pose_stabilizers():
        Stabilizer = YinsKalmanFilteredHeadPoseCalculator.Stabilizer
        stabilizers = []
        for _ in range(6):
            stabilizers.append(Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) )
        return stabilizers

    @staticmethod
    def __get_full_model_points(filename):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        # model_points *= 4
        model_points[:, -1] *= -1
        return model_points
    
    def __init__(self, face_model_path = None, inputFramesize = (1920, 1080), *args, **kwargs):
        super().__init__(*args, **kwargs)
        if face_model_path == None:
            face_model_path = CV2Res10SSD_frozen_face_model_path
        self._faceModelPoints = self.__get_full_model_points(face_model_path)
        self._front_depth = 100
        self._rectCorners3D = self._get_3d_points(rear_size = 80, rear_depth = 0, 
                                                  front_size = 10, front_depth = self._front_depth)
        
        # Camera internals
        self._camera_matrix = self._getCameraMatrix(inputFramesize)

        # Assuming no lens distortion
        self._dist_coeffs = np.array([[0.2562583722261407293], [-0.5884400171468063823], 
                                      [0.001658348839202715592], [-0.0006434617243149612104]
                                      ,[0.3660073010818283845]])

        # Rotation vector and translation vector
        self._rotation_vector = np.array([[-0.0], [0.0], [-0.0]]) # None11
        #self._rotation_vector = np.array([[-1.0, 0.0, 0], [0.0, 1.0, 0], [0.0, 0.0, -1.0]])
        self._translation_vector = np.array([[0.0], [0.0], [700.0]])# None 
        
        self._pose_stabilizers = self.__get_pose_stabilizers()

    def solve_pose_by_68_points(self, image_points): 
        image_points = image_points.astype('float32')
        #print('\r%s' % str(image_points.shape), end = '\r')
        (_, rotation_vector, translation_vector) = cv2.solvePnP(self._faceModelPoints,
                                                                image_points, self._camera_matrix, self._dist_coeffs#)
                                                    ,rvec=self._rotation_vector, tvec=self._translation_vector, useExtrinsicGuess=True)
        #self._rotation_vector, self._translation_vector = rotation_vector, translation_vector
        #tv = '%.2f %.2f %.2f' % tuple([t[0] for t in self._translation_vector])
        #rv = '%.2f %.2f %.2f' % tuple([math.degrees(t[0]) for t in self._rotation_vector])
        #print('\r[%s], [%s]' % (tv, rv), end = '\r')
        return (rotation_vector, translation_vector)

    def calculatePose(self, shape):
        pose = self.solve_pose_by_68_points(shape)
        rv =  np.array([math.degrees(t[0]) for t in self._rotation_vector])
        self._pose = np.concatenate((self._translation_vector[:,0], rv), 0)
        # Stabilize the pose.
        #stabile_pose = []
        #pose_np = np.array(pose).flatten()
        #for value, ps_stb in zip(pose_np, self._pose_stabilizers):
        #    ps_stb.update([value])
        #    stabile_pose.append(ps_stb.state[0])
        #rotation_vector, translation_vector = np.reshape(stabile_pose, (-1, 3))
        ## calc euler angle
        #rotation_mat, _ = cv2.Rodrigues(rotation_vector)
        #pose_mat = cv2.hconcat((rotation_mat, translation_vector))
        #cameraMatrix, rotMatrix, transVect, rotMatrixX, \
        #rotMatrixY, rotMatrixZ, pose = cv2.decomposeProjectionMatrix(pose_mat)
        ##tv = '%.2f %.2f %.2f %.2f' % tuple([t[0] for t in transVect])
        ##rv = '%.2f %.2f %.2f' % tuple([math.degrees(t[0]) for t in pose])
        ##print('\r[%s], [%s]' % (tv, rv), end = '\r')
        #self._pose[0] = pose[0] * (-1)
        #self._pose[1] = pose[1]
        #self._pose[2] = (pose[2] - 180) if pose[2] > 0 else pose[2] + 180
        #self._pose = self._pose.reshape((3,))
        #self._pose = pose
        return self._pose

class PoseEstimator(HeadPoseEstimatorABC):
    def __init__(self, faceDetector = None, landmarkDetector = None, poseCalculator = None, face_landmark_path = None, inputFramesize = (1920, 1080), *args, **kwargs):
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
        self._headPose3D = self.calculateHeadPose(frame)#[1][:2]
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
        #self._translation_vector = np.array([[-195.0], [-55.0], [0.0]])
        #self._translation_vector = np.array([[3.0], [0.0], [-700.0]])
        #self._translation_vector = np.array([[-14.97821226], [-10.62040383], [-120]])#-2053.03596872
        
        self._front_depth = 500
        self._rectCorners3D = self._get_3d_points(rear_size = 40, rear_depth = 0, 
                                                  front_size = 40, front_depth = self._front_depth)
        self._objectPointsVec = [self._faceModelPoints]
        self._imagePointsVec = []

    def calibrateCamera(self, imagePoints):
        ip = imagePoints.astype('float32')        
        print(imagePoints)
        self._imagePointsVec.append(ip)
        n = 7
        if len(self._imagePointsVec) < n+1:
            return
        self._imagePointsVec.pop(0)
        print(ip.shape, self._faceModelPoints.shape, len(self._objectPointsVec), len(self._imagePointsVec))
        retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(self._objectPointsVec, self._imagePointsVec, (1920, 1080), 
                                                                             self._camera_matrix, self._dist_coeffs,
                                                                             flags=(cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT))
        self._camera_matrix, self._dist_coeffs = cameraMatrix, distCoeffs
        self._rotation_vector, self._translation_vector = rvecs[0], tvecs[0]

    def calculateGaze(self):
        rc = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        
    def calculateEdges(self):
        ps3D = self._get_3d_points(rear_size = 200, rear_depth = 200, 
                                                  front_size = 200, front_depth = 0)
        
    def _get_points(self, rear_w, rear_h, rear_depth, front_w, front_h, front_depth):
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

    def calculateProjectionPointsAsGaze(self, shape, recalculatePose = False):
        if recalculatePose:
                self.calculatePose(shape)
        if not (self._rotation_vector is None or self._translation_vector is None):
            #rc = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).T
            #tc = np.array([[0.0], [0.0], [0.0]])
            #f = 2667.497359647048143
            #fd = 383/1920*f
            #ps3D = self._get_points(rear_w = 192.5, rear_h = 102, rear_depth = 1200, 
            #                                      front_w = 192.5, front_h = 106, front_depth = 276)
            #point_2d, _ = cv2.projectPoints(ps3D, rc, tc, self._camera_matrix,
            #                               self._dist_coeffs)
            tv = '%.2f %.2f %.2f' % tuple([t[0] for t in self._translation_vector])
            rv = '%.2f %.2f %.2f' % tuple([math.degrees(t[0]) for t in self._rotation_vector])
            self._front_depth = self._translation_vector[2, 0] #+  02667
            #print('\r%.2f, [%s], [%s]' % (self._front_depth, tv, rv), end = '\r')
            #print(self._front_depth)
            #self._rectCorners3D = self._get_3d_points(rear_size = 73, rear_depth = -55, 
            #                                          front_size = 40, front_depth = self._front_depth)
            #self._rectCorners3D = self._get_points(73, 77, -55, 40, 40, self._front_depth)
            self._rectCorners3D = self._get_3d_points(rear_size = 0, rear_depth = 55, 
                                        front_size = 0, front_depth = self._front_depth)
            point_2d, _ = cv2.projectPoints(self._rectCorners3D, self._rotation_vector, 
                                            self._translation_vector, self._camera_matrix,
                                           self._dist_coeffs)
            self._projectionPoints = np.int32(point_2d.reshape(-1, 2))
        return self._projectionPoints

    def calculateHeadGazeWithProjectionPoints(self, shape):
        self._pose = self.calculatePose(shape)
        self.calculateProjectionPointsAsGaze(shape)
        output = self._projectionPoints[-1, :]# - np.array([0, 1080/215*55+540])
        
        return output, self._projectionPoints

    def translateTo3D(self, points):
        rot = self._rotation_vector.copy()
        rot[0, 0] *= -1
        rot[1, 0] *= -1
        rotation_mat, _ = cv2.Rodrigues(rot)
        t_vec = self._translation_vector[:]
        #t_vec[1] *= -1
        project_mat = cv2.hconcat((rotation_mat, t_vec))
        project_mat = np.concatenate((project_mat, np.zeros((1, 4))), 0)
        project_mat[-1, -1] = 1
        points[:, -1] *= -1
        points_ = np.concatenate((points, np.ones((points.shape[0], 1))), 1)
        points3d = np.matmul(project_mat, points_.T).T[:, :-1]
        #print('\r%s %s %s' % (str(project_mat.shape), str(points_.shape), str(points3d.shape)), end = '\r')
        #cv2.undistortPoints(self._landmarks, self._camera_matrix, self._dist_coeffs)
        return points3d
    
    def get3DNose(self):
        nose =  self._get_3d_points(rear_size = 0, rear_depth = 55, 
                                    front_size = 0, front_depth = self._front_depth)
        return self.translateTo3D(nose)

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
        all3DPoints = np.concatenate((screen, landmarks3d, np.array([[200, 300, 0]]), nose))
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
        #nose = self.get3DNose()
        #return self.calculate3DProjection(nose)
        return self._projectionPoints

    def calculateall3DProjections(self):
        screenProj = self.calculate3DScreenProjection()
        landmarksProj = self.calculate3DLandmarksProjection()
        noseProj = self.calculate3DNoseProjection()
        return screenProj, landmarksProj, noseProj

class HeadGazer(PoseEstimator):
    def __init__(self, faceDetector = None, landmarkDetector = None, poseCalculator = None, face_landmark_path = None, inputFramesize = (1920, 1080), *args, **kwargs):
        if poseCalculator == None:
            poseCalculator = MuratcansHeadGazeCalculator(inputFramesize = inputFramesize)
        self._pPoints = np.zeros((1, 2))
        self._gazingFrameSize = inputFramesize
        self._halfFrameHeight = inputFramesize[1]/2
        super().__init__(faceDetector, landmarkDetector, poseCalculator, face_landmark_path, inputFramesize, *args, **kwargs)
        
    def calculateHeadPose(self, frame):
        self._landmarks = self._landmarkDetector.detectFacialLandmarks(frame)
        if len(self._landmarks) == 0:
            return self._headPose3D
        else:
            self._headPose3D = self._poseCalculator.calculatePose(self._landmarks)
            return self._headPose3D
                    
    def calculateHeadGaze(self, frame):
        self._landmarks = self._landmarkDetector.detectFacialLandmarks(frame)
        if len(self._landmarks) != 0:
            self._halfFrameHeight = frame.shape[0]/2
            g = self._poseCalculator.calculateHeadGazeWithProjectionPoints(self._landmarks) 
            self._headPose3D, self._pPoints = g
            return self._headPose3D
            
    def _calculateHeadPoseWithAnnotations(self, frame):
        self._headPose3D = self.calculateHeadGaze(frame)
        return self._headPose3D, self._pPoints, self._landmarks
    
    def getGazingFrameDimensions(self):
        #return int(1920), int(self._halfFrameHeight + 1080)
        #print(self._gazingFrameSize)
        return self._gazingFrameSize