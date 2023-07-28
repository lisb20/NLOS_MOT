import matplotlib.pyplot as plt
import pupil_apriltags as apriltag
import cv2
import numpy as np
from tqdm import tqdm
import pickle
from collections import defaultdict
import os

# Imat = [[1338.56729845250, 0, 0],
#         [0, 1323.97312268293, 0],
#         [942.273221383391, 473.805089526029, 1]]
# Imat = np.array(Imat).T

Imat = [[1131.57817294016,0,0],
         [0,1131.25028255906,0],
         [962.248182523716,543.533259657236,1]]
Imat = np.array(Imat).T




# Dmat = [-0.346167416278953, 0.110363006733750, 0, 0]
Dmat = [0.117006120213299,-0.126019080006247,0,0]
Dmat = np.array(Dmat)
# cam1_loc = [2960,6930]
cam1_loc = [1730,4340]  ## 1102: x为横向，y为纵向

Imat2 = [[1124.57531450059,0,0],
[0,1126.18079389101,0],
[968.764721434572,537.149978277261,1]]
Imat2 = np.array(Imat2).T  

Dmat2 = [0.110755756890598,-0.121416360832669,0,0]
Dmat2 = np.array(Dmat2)
# cam2_loc = [3150,160]  ### 有海康标的那个
cam2_loc = [1730,110]


npeo = 5

at_detector = apriltag.Detector(decode_sharpening=0.5, quad_decimate=1)


def pnp(obj, corners, C_M, d):
    '''
            obj: tag四个角的物理坐标,
            corners： tag四个角的像素坐标
            c_m:内参
            d:畸变矩阵
            注意：这四个都要是np.float_64

            输出： p_o：机器人物理坐标
            dir：机器人面对的方向
    '''
    s, r, t = cv2.solvePnP(obj, corners, C_M, d)
    r_m, ja = cv2.Rodrigues(r)

    robot_pos = np.array([0, 0, 0]).reshape((3, 1))
    robot_dir = np.array([0, 0, 100]).reshape((3, 1))

    p_o = np.linalg.pinv(r_m) @ (robot_pos - t)

    return p_o, t

halflen = 80.5 ##tag边长  
obj = np.array([[0, halflen, -halflen], [0, halflen, halflen], [0, -halflen, halflen],
               [0, -halflen, -halflen]], dtype=np.float64)


# read demo.mp4
folder = './1102/004/'
dirs = os.listdir(folder)
if ("loc1.pkl" in dirs) or ("loc2.pkl" in dirs):
    print("Warning overwrite loc1.pkl or loc2.pkl")
    # assert False


cap1 = cv2.VideoCapture(folder + 'output1.mp4')  ## cap1 约定为可以照到拍手的摄像头
cap2 = cv2.VideoCapture(folder + 'output2.mp4')



def get_location(cap,I,D):
    frame = 0
    fail_cnt = 0
    location = []
    for _ in range(npeo):
        location.append([])

    while(cap.isOpened()):
        ret, img = cap.read()
        frame += 1
        if frame % 100 == 0:
            print(frame)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        if ret == False:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        tags = at_detector.detect(gray)

        tag_num = len(tags)
        if(tag_num == 0):
            for p in location:
                p.append([None, None])
            fail_cnt += 1
            continue
        else:
            x = defaultdict(list)
            z = defaultdict(list)
            for tag in tags:
                peo_idx = tag.tag_id // 4
                corners = np.array(tag.corners, dtype=np.float64)
                p1, p2 = pnp(obj, corners, I, D)
                x[peo_idx].append(p2[0])
                z[peo_idx].append(p2[2])

        for i in range(npeo):
            if len(x[i]) == 0:
                location[i].append([None, None])
            else:
                location[i].append([np.mean(x[i]), np.mean(z[i])])

        if frame < 300:
            print(frame,location[0][frame-1])

    print('frame:', frame)
    print('fail_cnt:', fail_cnt)
    location = np.array(location)
    return location

def fix_frame(l1, l2):
    ## 补足缺失的帧
    loc = []
    print(l1.shape, l2.shape)
    for i in range(min(len(l1), len(l2))):
        if l1[i,0] is None and l2[i,0] is None:
            loc.append(None)
        elif l1[i,0] is None:
            loc.append(l2[i])
        elif l2[i,0] is None:
            loc.append(l1[i])
        else:
            loc.append([(l1[i,0]+l2[i,0])/2, (l1[i,1]+l2[i,1])/2])
        

    for idx, p in enumerate(loc):
        if p is None:
            front_id = -1
            back_id = -1
            for i in range(5):
                if idx-i < 0:
                    break
                if loc[idx-i] is not None:
                    front_id = idx-i
                    break
            for i in range(5):
                if idx+i >= len(loc):
                    break
                if loc[idx+i] is not None:
                    back_id = idx+i
                    break

            if front_id != -1 and back_id != -1:
                front = loc[front_id]
                back = loc[back_id]
                loc[idx] = [(front[0] * (back_id - idx) + back[0] * (idx - front_id)) / (back_id - front_id),
                                    (front[1] * (back_id - idx) + back[1] * (idx - front_id)) / (back_id - front_id)]
            elif front_id != -1:
                loc[idx] = loc[front_id]
            elif back_id != -1:
                loc[idx] = loc[back_id]
            elif idx > 100 and idx < len(loc) - 100:
                print("critical error: too many missing frames")
                assert False
            else:
                loc[idx] = [0, 0]
        else:
            pass
    
    loc = np.array(loc)
    # print('frame:', len(loc))

    x = loc[:, 0]
    y = loc[:, 1]
    ## x,y 通过低通滤波

    x = np.array(x)
    y = np.array(y)
    x = np.convolve(x, np.ones(10)/10, mode='same')
    y = np.convolve(y, np.ones(10)/10, mode='same')
    return x, y

### main


# loc1 = get_location(cap1,Imat,Dmat)
loc1 = pickle.load(open(folder + 'loc1.pkl', 'rb'))
pickle.dump(loc1, open(folder + 'loc1.pkl', 'wb'))
print('loc1:', loc1.shape)
for i in range(loc1.shape[1]):
    for peo in range(npeo):
        if loc1[peo,i,0] is not None:
            loc1[peo,i,0] =  cam1_loc[0] - loc1[peo,i,0]  ## 正负号由位置定
            loc1[peo,i,1] =  cam1_loc[1] - loc1[peo,i,1]

# loc2 = get_location(cap2,Imat2,Dmat2)
loc2 = pickle.load(open(folder + 'loc2.pkl', 'rb'))
pickle.dump(loc2, open(folder + 'loc2.pkl', 'wb'))
print('loc2:', loc2.shape)
for i in range(loc2.shape[1]):
    for peo in range(npeo):
        if loc2[peo,i,0] is not None:
            loc2[peo,i,0] =  cam2_loc[0] + loc2[peo,i,0]
            loc2[peo,i,1] =  cam2_loc[1] + loc2[peo,i,1]

plt.plot(loc1[0,:,0], loc1[0,:,1],"r")
plt.plot(loc2[0,:,0], loc2[0,:,1],"g")
plt.show()
# exit()


x = []
y = []
for i in range(npeo):
    if not all(loc1[i,:,0] == None) and not all(loc2[i,:,0] == None):
        xx,yy = fix_frame(loc1[i,:,:], loc2[i,:,:])
        x.append(xx)
        y.append(yy)

x = np.array(x)
y = np.array(y)

print(x.shape, y.shape)
from scipy.io import savemat
savemat(folder + 'loc.mat', {'x':x, 'y':y})

npeo = x.shape[0]
# print(x.shape, y.shape)
print("valid people:", npeo)
for i in range(npeo):
    plt.plot(x[i,10:-10], y[i,10:-10])
## x,y分度值250
plt.xticks(np.arange(0, 5500, 250))
plt.yticks(np.arange(0, 5500, 250))
plt.grid()
plt.show()


# # plt.show()
# plt.savefig('demo.png')


