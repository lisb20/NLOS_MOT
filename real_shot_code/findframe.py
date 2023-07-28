import cv2

## p1: 109-957   camera: 133-1994
fp = "./1102/005/output1.mp4"
#./1/001/MVI_0075.mp4    cam：220-9873   output1: 315-4810
#./1/002/MVI_0076.mp4    cam：74-19697   output1: 254-9219
#./1/003/MVI_0078.mp4    cam：68-27497    p1: 169-12686
#./1/004/MVI_0079.mp4    cam：77-12295    p1: 200-5188
#./1/005/MVI_0080.mp4    cam：117-44790    p1: 344-20081
#./1/006/MVI_0081.mp4    cam：57-38461    p1: 250-17276

#./1104/003/MVI_0085.mp4  cam：-  p1: 284-16050  作废
#./1104/007/MVI_0092.mp4  cam：58-40848  p1: 172-19907
#./1104/008/MVI_0093.mp4  cam：45-41784  p1: 215-20626
#./1104/009/MVI_0094.mp4  cam：74-42000  p1: 131-20816

#./1102/001/MVI_0095.mp4   cam：46-39995   p1: 204-19514
#./1102/003/MVI_0096.mp4   cam：37-41545   p1: 207-34292
#./1102/004/MVI_0097.mp4   cam：100-41688   p1: 241-35427
#./1102/005/MVI_0098.mp4   cam：94-41023  p1: 377-38379

cap = cv2.VideoCapture(fp)
cnt = 0
while (cap.isOpened()):
    ret, frame = cap.read()           
    if ret == True:
        
        cnt += 1
        if cnt % 100 == 0:
            print(cnt)  
        if cnt > 38300 :
            cv2.imshow('frame', frame)
            cv2.waitKey(0)
            print(cnt)
        else:
            pass
            # cv2.waitKey(1)
            # print(cnt)
    else:
        print(cnt)
        assert False
        break