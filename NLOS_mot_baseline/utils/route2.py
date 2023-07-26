import os
import time
# from collections import namedtuple

import numpy as np
from numpy import ndarray
from scipy.io import savemat, loadmat
from shapely.geometry import Polygon, Point

# from utils.vis import draw_route


class route_generator_mot(object):
    def __init__(self, map_size, forbidden_rate: float = 0.1, n_peo: int = 1):
        self.n_peo = n_peo
        self.forbidden_rate = forbidden_rate
        self.route_length = None
        self.map_size = map_size

        x_min, x_max, y_min, y_max = (map_size[0] * self.forbidden_rate,
                                      map_size[0] * (1 - self.forbidden_rate),
                                      map_size[1] * self.forbidden_rate,
                                      map_size[1] * (1 - self.forbidden_rate))
        self.boundary = Polygon(((x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)))

        self.e_position = None
        self.e_route = None
        self.c_route = None
        self.velocities = None
        ## 多人的情况
        self.e_route_all = []
        self.c_route_all = []
        self.v_all = [] 
        # 0.035m pre frame
        self.v_range = (0.03, 0.04)

    def _init_pv(self):
        bounds = self.boundary.bounds
        x = bounds[0] + (bounds[2] - bounds[0]) * np.random.rand()
        y = bounds[1] + (bounds[3] - bounds[1]) * np.random.rand()
        e_position = np.array([x, y])
        velocity = np.random.rand(2).astype(np.float32) - 0.5
        velocity = 0.035 * velocity / np.linalg.norm(velocity)

        return e_position, velocity

    def generate_route(self,
                       route_length: int = 256,
                       turn_rate: float = 0.15,
                       verbose: bool = False):
        self.route_length = route_length
        
        for i in range(self.n_peo):
            ep, v = self._init_pv()
            self.e_route_all.append([ep])
            self.v_all.append([v])
        
        for i in range(self.route_length):
            self.next_step(turn_rate)

        # print(len(self.e_route_all), len(self.e_route_all[0]))

        for i in range(self.n_peo):
            c_route = [(self.e_route_all[i][j] + self.e_route_all[i][j + 1]) * 0.5 for j in range(self.route_length)]
            c_route = np.stack(c_route)
            self.c_route_all.append(c_route / self.map_size)

        if verbose:
            print(len(self.velocities), len(self.c_route))
            print('velocities\n', np.stack(self.velocities))
            print('route:\n', np.stack(self.c_route))

    def next_step(self, turn_rate: float):
        v_frame = []
        for i in range(self.n_peo):
            e_pos = self.e_route_all[i][-1]
            v = self.v_all[i][-1]
            e_pos, v = self.check_boundary(e_pos, v)
            e_pos, v = self.check_collision(e_pos, v, i)
            v_frame.append(v)
        
        for i in range(self.n_peo):
            self.e_route_all[i].append(self.e_route_all[i][-1] + v_frame[i])
            delta_v = np.random.rand(2).astype(np.float32) - 0.5
            delta_v /= np.linalg.norm(delta_v)
            v_norm = self.v_range[0] + (self.v_range[1] - self.v_range[0]) * np.random.rand()

            v_frame[i] += turn_rate * v_norm * delta_v
            v_frame[i] *= v_norm / np.linalg.norm(v_frame[i])
            self.v_all[i].append(v_frame[i])

    def check_boundary(self, epos, v):
        point = Point(epos)
        if not self.boundary.contains(point):
            for i in range(2):
                p = epos[i]
                bound = self.boundary.bounds[i::2]
                if p < min(bound) or p > max(bound):
                    v[i] *= -1
                    break
                
        return epos, v

    def check_collision(self,epos,v,nowidx):
        try_pos = epos + v
        try_point = Point(try_pos)
        point_cnt = 0
        conflict_idx = []
        for i in range(self.n_peo):
            if i <= nowidx:
                continue
            p = Point(self.e_route_all[i][-1])
            if p.distance(try_point) < 0.3:
                point_cnt += 1
                conflict_idx.append(i)
        if point_cnt > 0:
            pass
            # print("conflict at: ", epos)


        if point_cnt == 1:
            ## 假设发生弹性碰撞
            v1 = np.array(self.v_all[nowidx][-1])
            v2 = np.array(self.v_all[conflict_idx[0]][-1])
            ## 旋转使得v1,v2角度大于45度
            if np.dot(v1, v2)/(np.linalg.norm(v1) * np.linalg.norm(v2)) < 0.707:
                return epos, self.v_all[conflict_idx[0]][-1]
            else:
                x1 = self.e_route_all[nowidx][-1]
                x2 = self.e_route_all[conflict_idx[0]][-1]
                v1 = np.linalg.norm(v1) * (x1 - x2) / np.linalg.norm(x1 - x2)
                v2 = np.linalg.norm(v2) * (x2 - x1) / np.linalg.norm(x2 - x1)
                return epos, v1
            
        elif point_cnt > 1:
            ## 三个人撞一块了，全部掉头
            return epos, -1*v
        else:
            return epos, v
            

    def draw_route(self, cmap: str = 'viridis', normalize: bool = True):
        route = np.stack(self.c_route) / np.array(self.map_size)
        map_size = np.array((1, 1)) if normalize else self.map_size
        draw_route(map_size, route, cmap, return_mode=None)

    def save_route(self, save_root: str, verbose: bool = False):

        time.sleep(1)
        save_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_dir = os.path.join(save_root, save_time)
        os.makedirs(save_dir, exist_ok=True)
        mat_path = os.path.join(save_dir, 'route.mat')
        map_size = np.array(self.map_size)
        # print(map_size)
        save_dict = {"map_size": map_size,
                     "route": np.stack(self.c_route) / map_size,  # (T, 2)
                     "velocities": np.stack(self.velocities[:-1])}

        savemat(mat_path, save_dict)
        if verbose:
            print(f'Save data into {mat_path} successfully!')

    def load_route(self,
                   mat_name: str,
                   save_dir: str):
        if not os.path.exists(save_dir):
            print(f"{save_dir} doesn't exist!")
        if mat_name is None:
            mat_names = sorted([f for f in os.listdir(save_dir) if f.endswith('.mat')])
            mat_name = mat_names[-1]
        mat_path = os.path.join(save_dir, mat_name)
        save_dict = loadmat(mat_path)

        self.e_route = [p for p in save_dict['route']]
        self.velocities = [v for v in save_dict['velocities']]
        print(f'Load data from {mat_path} successfully!')


def fix_real_trajectory(route_clip: ndarray, threshold: int = 10):
    bad_frames = find_miss_points(route_clip)
    b_len = len(bad_frames)
    counter = 1
    for i, frame in enumerate(bad_frames):
        if frame != 0:
            if i < b_len - 1 and bad_frames[i + 1] == frame + 1:
                counter += 1
            else:
                if counter <= threshold:
                    div = counter + 1
                    start_frame, end_frame = frame - counter, frame + 1
                    for j in range(1, div):
                        idx = start_frame + j
                        route_clip[idx] = route_clip[start_frame] * (j / div) \
                                          + route_clip[end_frame] * (1 - j / div)
                counter = 1

    return route_clip


def find_miss_points(route_clip: ndarray):
    return np.argwhere(route_clip == 0)[::2, 0]
