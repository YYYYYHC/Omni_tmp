import os
import numpy as np
from tqdm import tqdm
from utils_abs.track_cubes import track_cubes
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain
import torch
from utils_abs.sample import sample_all_abstraction
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import argparse
import threading

def process_cube(cubes, idxs, position):
    return track_cubes(cubes, idxs, position)

def process_cubes_with_threads(all_cubes, all_idxs, position, max_workers=4):
    """使用多线程处理单个文件内的所有cubes"""
    results = []
    
    # 创建线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有cube处理任务
        futures = [executor.submit(process_cube, cubes, idxs, position) 
                  for cubes, idxs in zip(all_cubes, all_idxs)]
        
        # 收集结果
        for future in as_completed(futures):
            results.append(future.result())
    
    return results

def process_single_file(file, data_source, vec_dir, save_dir, use_threading=False, thread_workers=4, mean=None, std=None):
    """处理单个文件的函数，支持多进程+多线程"""
    try:
        abstraction_dict_path = os.path.join(data_source, file)
        vec_path = os.path.join(vec_dir, file)
        abstraction_dict = np.load(abstraction_dict_path, allow_pickle=True).item()
        vecs = np.load(vec_path)
        vecs = vecs * std + mean
        position = recover_from_ric(torch.from_numpy(vecs), t2m_kinematic_chain).numpy()
        res = [[], [], [], [], []]
        
        for center_mode in [1,2,3,4,5]:
            all_cubes, all_idxs = sample_all_abstraction(abstraction_dict, center_mode, max_num1=3, max_num2=2)
            
            if use_threading and len(all_cubes) > 1:
                # 使用多线程处理cubes
                cube_results = process_cubes_with_threads(all_cubes, all_idxs, position, thread_workers)
                res[center_mode-1].extend(cube_results)
            else:
                # 单线程处理cubes
                for cubes, idxs in zip(all_cubes, all_idxs):
                    abstraction_full = process_cube(cubes, idxs, position)
                    res[center_mode-1].append(abstraction_full)
        
        outer = np.empty(len(res), dtype=object)     # 一级
        for i, sub in enumerate(res):
            inner = np.empty(len(sub), dtype=object) # 二级
            for j, arr in enumerate(sub):
                inner[j] = arr           # 确保到底层是 ndarray
            outer[i] = inner
        
        np.save(os.path.join(save_dir, file), outer, allow_pickle=True)
        return file, True
    except Exception as e:
        return file, str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理立方体抽象数据集（支持多进程+多线程）')
    parser.add_argument('--data_source', type=str, default="/home/yhc/OmniControl/dataset/HumanML3D/abstractions/cubeV3",
                        help='立方体抽象数据源目录路径')
    parser.add_argument('--vec_dir', type=str, default="/home/yhc/OmniControl/dataset/HumanML3D/new_joint_vecs",
                        help='关节位置数据目录路径')
    parser.add_argument('--save_dir', type=str, default="/home/yhc/OmniControl/dataset/HumanML3D/abstractions/cubeV3_processed",
                        help='保存处理后数据的目录路径')
    parser.add_argument('--use_multi_process', action='store_true', default=False,
                        help='是否使用多进程处理（文件级别并行）')
    parser.add_argument('--use_threading', action='store_true', default=False,
                        help='是否使用多线程处理（cube级别并行）')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='并行处理的最大工作进程数，默认为CPU核心数')
    parser.add_argument('--thread_workers', type=int, default=4,
                        help='每个进程内的线程数，默认为4')
    args = parser.parse_args()

    mean_path = os.path.join(os.path.dirname(os.path.dirname(args.vec_dir)), "Mean.npy")
    std_path = os.path.join(os.path.dirname(os.path.dirname(args.vec_dir)), "Std.npy")
    mean = np.load(mean_path)
    std = np.load(std_path)
    assert os.path.exists(mean_path) and os.path.exists(std_path), f"Mean and Std file not found in {os.path.dirname(args.vec_dir)}"
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 获取所有需要处理的文件
    files = [f for f in os.listdir(args.data_source) if f.endswith('.npy')]
    
    print(f"处理模式: {'多进程+多线程' if args.use_multi_process and args.use_threading else '多进程' if args.use_multi_process else '多线程' if args.use_threading else '单进程'}")
    print(f"文件数量: {len(files)}")
    if args.use_multi_process:
        print(f"进程数: {args.max_workers or os.cpu_count()}")
    if args.use_threading:
        print(f"线程数: {args.thread_workers}")
    
    if args.use_multi_process:
        print(f"使用多进程处理 {len(files)} 个文件，工作进程数: {args.max_workers or os.cpu_count()}")
        
        # 创建偏函数，固定参数
        process_func = partial(process_single_file, 
                             data_source=args.data_source,
                             vec_dir=args.vec_dir,
                             save_dir=args.save_dir,
                             use_threading=args.use_threading,
                             thread_workers=args.thread_workers,
                             mean=mean,
                             std=std)
        
        # 使用ProcessPoolExecutor进行文件级别的多进程处理
        with ProcessPoolExecutor(max_workers=args.max_workers or os.cpu_count()) as executor:
            # 提交所有文件处理任务
            futures = [executor.submit(process_func, file) for file in files]
            
            # 使用tqdm显示进度
            results = []
            for future in tqdm(as_completed(futures), desc="处理文件", total=len(files)):
                file, result = future.result()
                results.append((file, result))
                
                if result is not True:
                    print(f"文件 {file} 处理失败: {result}")
                else:
                    print(f"文件 {file} 处理完成")
    else:
        print(f"使用单进程处理 {len(files)} 个文件")
        
        # 单进程处理
        for file in tqdm(files, desc="处理文件"):
            file, result = process_single_file(file, args.data_source, args.vec_dir, args.save_dir, 
                                            args.use_threading, args.thread_workers)
            if result is not True:
                print(f"文件 {file} 处理失败: {result}")
            else:
                print(f"文件 {file} 处理完成")