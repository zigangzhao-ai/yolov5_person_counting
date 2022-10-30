from cvs import *
import cv2
from utils import process_points

# # ### 保存图片 ###
# cap = cvs.VideoCapture("/home/lesson5_codes/aidlux/video.mp4")
# frame = cap.read()
# cv2.imwrite("image.jpg",frame)
# cap.release()
# cv2.destroyAllWindows()

### 显示监测区域 ###
cap = cvs.VideoCapture("/home/lesson5_codes/aidlux/video.mp4")
frame_id = 0
while True:
    frame = cap.read()
    if frame is None:
        continue
    # 绘制越界监测区域
    points = [[593,176],[904,243],[835,323],[507,259]]
    color_light_green=(144, 238, 144)  ##浅绿色
    res_img = process_points(frame,points,color_light_green)
    cvs.imshow(res_img)


    