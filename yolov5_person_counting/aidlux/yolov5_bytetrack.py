# aidlux相关
from cvs import *
import aidlite_gpu
from utils import detect_postprocess, preprocess_img, draw_detect_res, scale_coords,process_points, is_in_poly
import cv2
# bytetrack
from track.tracker.byte_tracker import BYTETracker
from track.utils.visualize import plot_tracking


# 加载模型
model_path = '/home/lesson3_aidlux/aidlux/yolov5n_best_zzg-fp16.tflite'
in_shape = [1 * 640 * 640 * 3 * 4]
out_shape = [1 * 25200 * 6 * 4]

# 载入模型
aidlite = aidlite_gpu.aidlite()
# 载入yolov5检测模型
aidlite.ANNModel(model_path, in_shape, out_shape, 4, 0)

tracker = BYTETracker(frame_rate=30)
track_id_status = {}
cap = cvs.VideoCapture("/home/lesson3_aidlux/aidlux/video.mp4")
frame_id = 0
while True:
    frame = cap.read()
    if frame is None:
        continue
    frame_id += 1
    if frame_id % 3 != 0:
        continue
     # 预处理
    img = preprocess_img(frame, target_shape=(640, 640), div_num=255, means=None, stds=None)
    # 数据转换：因为setTensor_Fp32()需要的是float32类型的数据，所以送入的input的数据需为float32,大多数的开发者都会忘记将图像的数据类型转换为float32
    aidlite.setInput_Float32(img, 640, 640)
    # 模型推理API
    aidlite.invoke()
    # 读取返回的结果
    pred = aidlite.getOutput_Float32(0)
    # 数据维度转换
    pred = pred.reshape(1, 25200, 6)[0]
    # 模型推理后处理
    pred = detect_postprocess(pred, frame.shape, [640, 640, 3], conf_thres=0.4, iou_thres=0.45)
    # 绘制推理结果
    res_img = draw_detect_res(frame, pred)

    # 目标追踪相关功能
    det = []
    # Process predictions
    for box in pred[0]:  # per image
        box[2] += box[0]
        box[3] += box[1]
        det.append(box)
    if len(det):
        # Rescale boxes from img_size to im0 size
        online_targets = tracker.update(det, [frame.shape[0], frame.shape[1]])
        online_tlwhs = []
        online_ids = []
        online_scores = []
        # 取出每个目标的追踪信息
        for t in online_targets:
            # 目标的检测框信息
            tlwh = t.tlwh
            # 目标的track_id信息
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            # 针对目标绘制追踪相关信息
            res_img = plot_tracking(res_img, online_tlwhs, online_ids, 0,0)
    cvs.imshow(res_img)