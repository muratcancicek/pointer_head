sep = '\\'
DATASETS_Folder = 'C:\\cStorage\\Datasets' + sep
WhiteBallExpFolder = DATASETS_Folder + 'WhiteBallExp' + sep
MergedVideosFolder = WhiteBallExpFolder + 'MergedVideos' + sep
TrailsFolder = WhiteBallExpFolder + 'trails' + sep
TrailsDataFolder = TrailsFolder + 'data' + sep
TrailVideosFolder = TrailsFolder + 'videos' + sep
TrailSnapshotsFolder = TrailsFolder + 'snapshots' + sep
SubjectsFolder = WhiteBallExpFolder + 'Subjects' + sep
PostDataFolder = WhiteBallExpFolder + 'PostData_pnp_kf' + sep
AnalysisFolder = WhiteBallExpFolder + 'Analysis' + sep

TF_Models_Folder = DATASETS_Folder + 'TF_Models' + sep
TFMobileNetSSDFaceDetector_tf_model_path = TF_Models_Folder + \
    'TFMobileNetSSDFaceDetector' + sep + 'frozen_inference_graph_face.pb'
YinsFacialLandmarkDetector_tf_model_path = TF_Models_Folder + \
    'YinsCNNBasedFacialLandmarkDetector' + sep + 'frozen_inference_graph.pb'

CV2Res10SSD_frozen_Folder = DATASETS_Folder + 'CV2Nets' +\
   sep + 'CV2Res10SSD' + sep
CV2Res10SSD_frozen_face_model_path = CV2Res10SSD_frozen_Folder + \
    'face68_model.txt'
