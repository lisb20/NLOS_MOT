import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

project_folder = './data_1102/'
folder = './1102/001/'
map_size = [3.384,5]
cap = cv2.VideoCapture(folder +  "MVI_0095.mp4")
cnt = 0
start = 46
end = 39995

st_trace = 204
ed_trace = 19514

from scipy.io import loadmat
mat = loadmat(folder + 'loc.mat')
x = mat['x'].T.squeeze()
y = mat['y'].T.squeeze()

x = x[st_trace:ed_trace]
xx = x
print(x.shape)
x = x.astype(np.float32)
## 插值使得x为end-start+1个元素
x = np.interp(np.arange(end-start+1)/(end-start+1), np.arange(ed_trace-st_trace)/(ed_trace-st_trace), x)


y = y[st_trace:ed_trace]
y = y.astype(np.float32)
yy = y
y = np.interp(np.arange(end-start+1)/(end-start+1), np.arange(ed_trace-st_trace)/(ed_trace-st_trace), y)


x = x/map_size[0]/1000
y = y/map_size[1]/1000

assert all(x >= 0) and all(x <= 1.05)
assert all(y >= 0) and all(y <= 1.05)
# plt.plot(x[:100] * 1000 * map_size[0],y[:100] * 1000 * map_size[1])
# plt.plot(xx[:100],yy[:100]) 
# plt.show()

# exit()

vid_len = 256
idx = 0
nowvid = []

offset = len(os.listdir(project_folder))

frames = []
while (cap.isOpened()):
    ret, frame = cap.read()
    # frames.append(frame)
    if ret == True:
       cnt += 1
    else:
        # assert False
        break
    if cnt > start + 10 and cnt < end - 10:
        frames.append(frame)
        if len(frames) == vid_len:
            for i in frames:
                f = i
                f = cv2.resize(f, (128,128))
                # cv2.imshow('frame', f)
                # cv2.waitKey(0)

                ## c,h,w -> h,w,c
                f = f.transpose(2,0,1)
                nowvid.append(f)

                if len(nowvid) == vid_len:
                    out_folder = os.path.join(project_folder, str(idx + offset))
                    if not os.path.exists(out_folder):
                        os.makedirs(out_folder)
                    nowvid = np.array(nowvid)
                    nowvid = np.transpose(nowvid, (1,0,2,3))
                    # print(nowvid.shape)
                    assert nowvid.shape == (3, vid_len, 128, 128)
                    np.save(out_folder + '/video_128.npy', nowvid)
                    
                    nowvid = []
                    idxstart = cnt - vid_len - start
                    idxend = cnt - start
                    ## cnt:相机开始后的绝对帧数
                    ## x,y: 长度为start-end

                    pos = [x[idxstart:idxend], y[idxstart:idxend]]
                    v = [x[idxstart+1:idxend+1] - x[idxstart:idxend], y[idxstart+1:idxend+1] - y[idxstart:idxend]]
                    pos = np.array(pos).T
                    v = np.array(v).T
                    assert pos.shape == (vid_len, 2)
                    assert v.shape == (vid_len, 2)
                    # print(pos)
                    from scipy.io import savemat
                    savemat(out_folder + "/route.mat", {"map_size":map_size,'route':pos, 'velocities':v})

                    idx += 1
                    print(idx)

            frames = []