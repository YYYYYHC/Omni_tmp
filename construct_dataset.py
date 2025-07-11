import os
import numpy as np
from tqdm import tqdm
from utils_abs.track_cubes import track_cubes
from utils_abs.sample import sample_all_abstraction
from concurrent.futures import ProcessPoolExecutor
from functools import partial

data_source = "/home/yhc/OmniControl/dataset/HumanML3D/abstractions/cubeV2"
position_dir = "/home/yhc/OmniControl/dataset/HumanML3D/new_joints"

save_dir = "/home/yhc/OmniControl/dataset/HumanML3D/abstractions/cubeV2_processed"

def process_cube(cubes, idxs, position):
    return track_cubes(cubes, idxs, position)

if __name__ == "__main__":
    save_dir = "/home/yhc/OmniControl/dataset/HumanML3D/abstractions/cubeV2_processed"
    os.makedirs(save_dir, exist_ok=True)
    USE_MULTI_PROCESS = False
    NORMALIZE = False
    if NORMALIZE:
        save_dir = "/home/yhc/OmniControl/dataset/HumanML3D/abstractions/cubeV2_processed_normalized"
        mean = np.load('/home/yhc/OmniControl/dataset/humanml_spatial_norm/Mean_raw.npy')
        std = np.load('/home/yhc/OmniControl/dataset/humanml_spatial_norm/Std_raw.npy')

        mean = mean.reshape(-1,3)
        std = std.reshape(-1,3)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file in tqdm(os.listdir(data_source), desc="处理文件", total=len(os.listdir(data_source))):
        abstraction_dict_path = os.path.join(data_source, file)
        position_path = os.path.join(position_dir, file)
        abstraction_dict = np.load(abstraction_dict_path, allow_pickle=True).item()
        position = np.load(position_path)
        if NORMALIZE:
            position = (position - mean) / std
        res = [[], [], [], []]
        for center_mode in [1,2,3,4]:
            all_cubes, all_idxs = sample_all_abstraction(abstraction_dict, center_mode, max_num1=2, max_num2=1)
            if USE_MULTI_PROCESS:
                print("using multi process")
                # 创建一个偏函数，固定position参数
                process_func = partial(process_cube, position=position)
                    
                # 使用ProcessPoolExecutor进行多进程处理
                with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                    # 提交所有任务
                    futures = [executor.submit(process_func, cubes, idxs) for cubes, idxs in zip(all_cubes, all_idxs)]
                    
                    # 使用tqdm显示进度
                    for future in tqdm(futures, desc=f"处理中心模式 {center_mode}", total=len(all_cubes)):
                        abstraction_full = future.result()
                        res[center_mode-1].append(abstraction_full)
            else:
                print("using single process")
                for cubes, idxs in zip(all_cubes, all_idxs):
                    abstraction_full = process_cube(cubes, idxs, position)
                    res[center_mode-1].append(abstraction_full)
        outer = np.empty(len(res), dtype=object)     # 一级
        for i, sub in enumerate(res):
            inner = np.empty(len(sub), dtype=object) # 二级
            for j, arr in enumerate(sub):
                inner[j] = arr           # 确保到底层是 ndarray
            outer[i] = inner
        np.save(os.path.join(save_dir, file), outer,allow_pickle=True)
        exit()
        
