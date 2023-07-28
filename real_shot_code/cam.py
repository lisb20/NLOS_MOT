import cv2
import os
## 同时开启两个摄像头并录像
# cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(1)
# cap2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(2)

print(cap1.isOpened())
print(cap2.isOpened())
print(cap1.get(cv2.CAP_PROP_FPS))
print(cap2.get(cv2.CAP_PROP_FPS))

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out1 = cv2.VideoWriter('output1.avi', fourcc, 30.0, (1920, 1080))
# out2 = cv2.VideoWriter('output2.avi', fourcc, 30.0, (1920, 1080))
fp = "./1102/005/"
if os.path.exists(fp):
    dirs = os.listdir(fp)
    assert len(dirs) == 0 ## 确保文件夹为空
else:
    os.makedirs(fp)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out1 = cv2.VideoWriter(fp + 'output1.mp4', fourcc, 30.0, (1920, 1080))
out2 = cv2.VideoWriter(fp + 'output2.mp4', fourcc, 30.0, (1920, 1080))

cnt = 0
while (cap1.isOpened() and cap2.isOpened()):
    cnt += 1
    if cnt % 100 == 0:
        print(cnt)
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 == True and ret2 == True:
        out1.write(frame1)
        out2.write(frame2)
        cv2.imshow('frame1', frame1)
        cv2.imshow('frame2', frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap1.release()
cap2.release()
