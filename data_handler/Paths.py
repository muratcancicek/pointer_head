sep = '\\'
DATASETS_Folder = 'C:\\cStorage\\Datasets' + sep
WhiteBallExpFolder = 'WhiteBallExp' + sep
MergedVideosFolder = WhiteBallExpFolder + 'MergedVideos' + sep
TrailsFolder = WhiteBallExpFolder + 'trails' + sep
TrailsDataFolder = TrailsFolder + 'data' + sep
TrailVideosFolder = TrailsFolder + 'videos' + sep
TrailSnapshotsFolder = TrailsFolder + 'snapshots' + sep
SubjectsFolder = WhiteBallExpFolder + 'Subjects' + sep

TF_Models_Folder = DATASETS_Folder + 'TF_Models' + sep
TFMobileNetSSDFaceDetector_tf_model_path = TF_Models_Folder + \
    'TFMobileNetSSDFaceDetector' + sep + 'frozen_inference_graph_face.pb'
