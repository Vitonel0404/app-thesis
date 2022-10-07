import torch
import cv2
import numpy as np

model=torch.hub.load('ultralytics/yolov5','custom',path='/home/robertoboss/Desktop/objectDetect/app/runs/train/exp/weights/best.pt')

#videocaptura

cap=cv2.VideoCapture(0)

while True:
    #lectrua de videocaputra
    ret, frame = cap.read()

    #realizar detecciones
    detect = model(frame)

    #mostrar fps
    cv2.imshow("detector de palomas",np.squeeze(detect.render()))

    t=cv2.waitKey(5)
    if t==27:
        break

cap.release()
cv2.destroyAllWindows()
#$ pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
