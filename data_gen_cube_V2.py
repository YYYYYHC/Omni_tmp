
from utils.plot_script import plot_3d_motion_with_cubes, plot_3d_motion_with_cubes_static
from utils.paramUtil import t2m_kinematic_chain
import numpy as np
import random
from scipy.spatial.transform import Rotation as R


import matplotlib 
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from yhcutils import  get_template_cube, line2cube, pcaCube
import os
import itertools

def generate_combinations(input_list, sample_num):
    # 生成从输入列表中采样 sample_num 个元素的所有可能组合
    return list(itertools.combinations(input_list, sample_num))
 
def get_cube_from_branches(joints, kinematic_chain):
    all_branches = []
    all_branches_idxs = []
    joint = joints[0] # (22,3)
    # print(kinematic_chain)
    # no cross branch merging
    for branch in kinematic_chain:
        if len(branch) <= 1:
            continue
        branch_len = len(branch)
        sampls = []
        sampls_idxs = []
        # print(f"branch: {branch}")
        for sample_num in range(0, branch_len-1):
            if sample_num == 0:
                cubes = [pcaCube(joint[branch])]
                combs = [ cubes]
                combs_idxs =[[branch]]
                sampls.append(combs)
                sampls_idxs.append(combs_idxs)
            else:
                combinations = generate_combinations(list(range(1,branch_len-1)), sample_num)
                combs = []
                combs_idxs = []
                for comb in combinations:
                    #TBD: get cube from comb, append the cube to all_branches
                    cubes_idx = [ [i for i in range(0, comb[0]+1)] ] 
                                      
                    cubes_idx += [[j for j in range(comb[i], comb[i+1]+1)] for i in range(len(comb)-1)]
                    cubes_idx += [[i for i in range(comb[-1], branch_len-1 + 1)]]
                    # print(cubes_idx)
                    cubes = []
                    cubes_idxs_projected=[]
                    for cube_idx in cubes_idx:
                        if len(cube_idx) == 2:
                            p1 = cube_idx[0]
                            p2 = cube_idx[1]
                            template_cube = get_template_cube(S=np.array([0.05, 0.05, 1]))
                            cubes.append(line2cube(joint[branch[p1]][None, ...], joint[branch[ p2]][None, ...], template_cube[None,...])[0])
                            cubes_idxs_projected.append([branch[p1], branch[p2]])
                            # print(f"branch[p1]: {branch[p1]}")
                            # print(f"branch[p2]: {branch[p2]}")
                        else:
                            
                            points = joint[np.array(branch)[cube_idx]]
                            # 通过添加相邻点的中点使点集变得更加稠密
                            
                            # print(f"points.shape: {points.shape}")
                            
                            cube = pcaCube(points)
                            cubes.append(cube)
                            cubes_idxs_projected.append(np.array(branch)[cube_idx].tolist())
                    combs.append(cubes)
                    combs_idxs.append(cubes_idxs_projected)
                sampls.append(combs)
                sampls_idxs.append(combs_idxs)
        all_branches.append(sampls)
        all_branches_idxs.append(sampls_idxs)
    return all_branches, all_branches_idxs
def get_abstract_cube(joints, kinematic_chain, center_mode = 0):
    # extract an abstaction from joints given kinematic_chain
    included_idxs = []
    joint = joints[0] # (22,3)
    if center_mode == 0:
        all_branches, all_branches_idxs = get_cube_from_branches(joints, kinematic_chain)
        return all_branches, all_branches_idxs #(branch_num, sample_num,comb_num, cube_num, 8, 3)
    if center_mode == 1:
        # one extra branch: merge 0
        center_samples = []
        center_samples_idxs = []
        branch_samples = []
        branch_samples_idxs = []
        
        #leg joint numbers to be merged: 1 - 4; waist joint numbers to be merged: 1 - 3
        # 生成腿部关节和腰部关节的所有配对组合（笛卡尔积）,实际上这里生成的是sample层的可能组合，每个sample只有一种combination
        leg_sample_amount = [1, 2, 3, 4]
        waist_sample_amount = [1, 2, 3]
        leg_waist_sample_amount_combinations = list(itertools.product(leg_sample_amount, waist_sample_amount))

        for leg_waist_sample_amount in leg_waist_sample_amount_combinations:
            leg_amount = leg_waist_sample_amount[0]
            waist_amount = leg_waist_sample_amount[1]
            point_idxs = [0] + kinematic_chain[0][1:leg_amount+1] + kinematic_chain[1][1:leg_amount+1] + kinematic_chain[2][1:waist_amount+1]
            cube = pcaCube(joint[point_idxs][None,...][0])
            
            new_kinematic_chain = kinematic_chain.copy()
            new_kinematic_chain[0] = kinematic_chain[0][leg_amount:]
            new_kinematic_chain[1] = kinematic_chain[1][leg_amount:]
            new_kinematic_chain[2] = kinematic_chain[2][waist_amount:]
            branches, branches_idxs = get_cube_from_branches(joints, new_kinematic_chain)
            center_samples.append(cube)
            center_samples_idxs.append([point_idxs])
            branch_samples.append(branches)
            branch_samples_idxs.append(branches_idxs)
        return center_samples, branch_samples , center_samples_idxs, branch_samples_idxs
    if center_mode == 2:
        # one extra branch: merge 9
        center_samples = []
        center_samples_idxs = []
        branch_samples = []
        branch_samples_idxs = []
        #hand joint numbers to be merged: 1 - 4; head joint numbers to be merged: 1 - 2; waist joint numbers to be merged: 1 - 3
        # 生成手部关节和头部关节的所有配对组合（笛卡尔积）,实际上这里生成的是sample层的可能组合，每个sample只有一种combination
        hand_sample_amount = [1, 2, 3, 4]
        head_sample_amount = [0,1, 2]
        waist_sample_amount = [0,1, 2, 3]
        hand_head_waist_sample_amount_combinations = list(itertools.product(hand_sample_amount, head_sample_amount, waist_sample_amount))

        for hand_head_waist_sample_amount in hand_head_waist_sample_amount_combinations:
            hand_amount = hand_head_waist_sample_amount[0]
            head_amount = hand_head_waist_sample_amount[1]
            waist_amount = hand_head_waist_sample_amount[2]
            point_idxs = kinematic_chain[2][3-waist_amount:3+head_amount+1] + kinematic_chain[3][1:hand_amount+1] + kinematic_chain[4][1:hand_amount+1] 
            cube = pcaCube(joint[point_idxs][None,...][0])
            
            new_kinematic_chain = kinematic_chain.copy()
            new_kinematic_chain[3] = kinematic_chain[3][hand_amount:]
            new_kinematic_chain[4] = kinematic_chain[4][hand_amount:]
            new_kinematic_chain[2] = kinematic_chain[2][:3-waist_amount+1] 
            new_kinematic_chain.append(kinematic_chain[2][3+head_amount:])
            branches, branches_idxs = get_cube_from_branches(joints, new_kinematic_chain)
            center_samples.append(cube)
            center_samples_idxs.append([point_idxs])
            branch_samples.append(branches)
            branch_samples_idxs.append(branches_idxs)
        return center_samples, branch_samples, center_samples_idxs, branch_samples_idxs
    if center_mode == 3:
        # one extra branch: merge 0 and 9 together
        center_samples = []
        center_samples_idxs = []
        branch_samples = []
        branch_samples_idxs = []
        #hand joint numbers to be merged: 1 - 4; head joint numbers to be merged: 1 - 2, leg joint numbers to be merged: 1 - 4
        # 生成手部关节和头部关节的所有配对组合（笛卡尔积）,实际上这里生成的是sample层的可能组合，每个sample只有一种combination
        hand_sample_amount = [1, 2, 3, 4]
        head_sample_amount = [0,1, 2]
        leg_sample_amount = [1, 2, 3, 4]
        hand_head_leg_sample_amount_combinations = list(itertools.product(hand_sample_amount, head_sample_amount, leg_sample_amount))

        for hand_head_leg_sample_amount in hand_head_leg_sample_amount_combinations:
            hand_amount = hand_head_leg_sample_amount[0]
            head_amount = hand_head_leg_sample_amount[1]
            leg_amount = hand_head_leg_sample_amount[2]
            point_idxs = kinematic_chain[2][:3+head_amount+1]\
                + kinematic_chain[3][1:hand_amount+1]\
                    + kinematic_chain[4][1:hand_amount+1] \
                        + kinematic_chain[0][1:leg_amount+1]\
                            + kinematic_chain[1][1:leg_amount+1]
                        

            cube = pcaCube(joint[point_idxs][None,...][0])
            
            new_kinematic_chain = kinematic_chain.copy()
            new_kinematic_chain[0] = kinematic_chain[0][leg_amount:]
            new_kinematic_chain[1] = kinematic_chain[1][leg_amount:]
            
            new_kinematic_chain[2] = kinematic_chain[2][3+head_amount:]
            
            new_kinematic_chain[3] = kinematic_chain[3][hand_amount:]
            new_kinematic_chain[4] = kinematic_chain[4][hand_amount:]
            
            branches, branches_idxs = get_cube_from_branches(joints, new_kinematic_chain)
            center_samples.append(cube)
            center_samples_idxs.append([point_idxs])
            branch_samples.append(branches)
            branch_samples_idxs.append(branches_idxs)
        return center_samples, branch_samples, center_samples_idxs, branch_samples_idxs
    if center_mode == 4:
        # two extra branches: merge 0 and 9 separately
        center_samples = []
        branch_samples = []
        center_samples_idxs = []
        branch_samples_idxs = []
        #hand joint numbers to be merged: 1 - 4; head joint numbers to be merged: 1 - 2; waist joint numbers to be merged: 1 - 3
        # leg joint numbers to be merged: 1 - 4
        # 生成手部关节和头部关节的所有配对组合（笛卡尔积）,实际上这里生成的是sample层的可能组合，每个sample只有一种combination
        hand_sample_amount = [1, 2, 3, 4]
        head_sample_amount = [0,1, 2]
        waist1_sample_amount = [0,1, 2, 3]
        waist2_sample_amount = [0,1, 2, 3]
        leg_sample_amount = [1, 2, 3, 4]
        hand_head_waist1_waist2_leg_sample_amount_combinations = list(itertools.product(hand_sample_amount, head_sample_amount, waist1_sample_amount, waist2_sample_amount, leg_sample_amount))

        for hand_head_waist1_waist2_leg_sample_amount in hand_head_waist1_waist2_leg_sample_amount_combinations:
            hand_amount = hand_head_waist1_waist2_leg_sample_amount[0]
            head_amount = hand_head_waist1_waist2_leg_sample_amount[1]
            waist1_amount = hand_head_waist1_waist2_leg_sample_amount[2]
            waist2_amount = hand_head_waist1_waist2_leg_sample_amount[3]
            leg_amount = hand_head_waist1_waist2_leg_sample_amount[4]
            if waist1_amount + waist2_amount > 3:
                continue
            point_idxs1 = kinematic_chain[2][3-waist1_amount:3+head_amount+1] + kinematic_chain[3][1:hand_amount+1] + kinematic_chain[4][1:hand_amount+1] 
            point_idxs2 = kinematic_chain[0][1:leg_amount+1] + kinematic_chain[1][1:leg_amount+1] +kinematic_chain[2][:waist2_amount+1]
            cube1 = pcaCube(joint[point_idxs1][None,...][0])
            cube2 = pcaCube(joint[point_idxs2][None,...][0])
            cube = np.concatenate([cube1, cube2], axis=0)
            
            new_kinematic_chain = kinematic_chain.copy()
            new_kinematic_chain[0] = kinematic_chain[0][leg_amount:]
            new_kinematic_chain[1] = kinematic_chain[1][leg_amount:]
            new_kinematic_chain[2] = kinematic_chain[2][waist2_amount:3-waist1_amount+1] 
            new_kinematic_chain[3] = kinematic_chain[3][hand_amount:]
            new_kinematic_chain[4] = kinematic_chain[4][hand_amount:]
            
            new_kinematic_chain.append(kinematic_chain[2][3+head_amount:])
            branches, branches_idxs = get_cube_from_branches(joints, new_kinematic_chain)
            center_samples.append(cube)
            center_samples_idxs.append([point_idxs1, point_idxs2])
            branch_samples.append(branches)
            branch_samples_idxs.append(branches_idxs)
        return center_samples, branch_samples, center_samples_idxs, branch_samples_idxs

def _add_normal(points):
    """
    给 (2,3) 的点集增补一个法向点，返回 (3,3)。
    生成规则保证：
    1. 新点与原两点不共线
    2. 距离量级与原连线相近
    """
    assert points.shape == (2, 3)
    p0, p1 = points
    d = p1 - p0
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-8:
        raise ValueError("两点重合，无法生成法向点")

    # 选一个与 d 不平行的参考轴
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(d / d_norm, ref)) > 0.9:    # 几乎共线
        ref = np.array([0.0, 1.0, 0.0])

    n = np.cross(d, ref)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:                          # 退化，再换一个轴
        ref = np.array([0.0, 0.0, 1.0])
        n = np.cross(d, ref)
        n_norm = np.linalg.norm(n)

    n = n / n_norm * (0.5 * d_norm)           # 缩放
    p2 = p0 + n
    return np.vstack([points, p2])             # (3,3)

def track_a_cube(cube, cube_idx, joints):
    '''
    given a cube and its relative joints' idx, track the cube based on the joints's motion
    
    return: a set of cubes following the joints' motion
    '''
    relative_joints = joints[:, cube_idx,:]
    # given the relative joints with shape (time, joint_num, 3), compute the 
    # relative motion based on rigid assumption
    # 计算每一帧的质心
    def compute_centroid(points):
        return np.mean(points, axis=0)

    # 初始化
    num_frames, m, _ = relative_joints.shape
    tracked_cubes = np.zeros((num_frames, 8, 3))

    # 初始立方体
    tracked_cubes[0] = cube

    # 逐帧计算刚体运动
    for i in range(1, num_frames):
        prev_cube = tracked_cubes[i-1]
        # 当前帧和前一帧的关节位置
        prev_joints = relative_joints[i-1]
        curr_joints = relative_joints[i]
        # 若只有两个点，给上一帧/当前帧各补一个法向点
        if m == 2:
            prev_joints = _add_normal(prev_joints)     # (3,3)
            curr_joints = _add_normal(curr_joints)
        # 计算质心
        prev_centroid = compute_centroid(prev_joints)
        curr_centroid = compute_centroid(curr_joints)

        # 去中心化
        prev_joints_centered = prev_joints - prev_centroid
        curr_joints_centered = curr_joints - curr_centroid

        # 使用SVD计算旋转矩阵
        H = prev_joints_centered.T @ curr_joints_centered
        U, S, Vt = np.linalg.svd(H)
        R_matrix = Vt.T @ U.T

        # 确保旋转矩阵是正交的
        if np.linalg.det(R_matrix) < 0:
            Vt[-1, :] *= -1
            R_matrix = Vt.T @ U.T

        # 计算平移
        translation = curr_centroid - R_matrix @ prev_centroid
        

        # 应用旋转和平移到立方体
        tracked_cubes[i] = (R_matrix @ prev_cube.T).T + translation
        # tracked_cubes[i] = cube + translation

    return tracked_cubes


def track_a_sample(all_branches, all_branches_idxs, center_samples, center_samples_idxs, joints, savename):
    # all_branches:
    # the input is under the same centering strategy, (branch_num: branches, sample_num: number of 
    # abs within the same branch, comb_num: given the number of abs, the number of joint combination,
    # cube_num: the number of cube in each combination, 8, 3)
    final_cubes = []
    final_cubes_idxs = []
    if len(all_branches) != 0:
        # get a random sample from every branch
        cube_branches = [[] for _ in range(len(all_branches))]
        cube_branches_idxs = [[] for _ in range(len(all_branches))]
        for branchid, branch in enumerate(all_branches):
            for sampleid, sample in enumerate(branch):
                for combid, comb in enumerate(sample):
                    cubes = []
                    cubes_idxs = []
                    for cubeid, cube in enumerate(comb):
                        cubes.append(cube)
                        cubes_idxs.append(all_branches_idxs[branchid][sampleid][combid][cubeid])
                    cube_branches[branchid].append(cubes)
                    cube_branches_idxs[branchid].append(cubes_idxs)
        # 从每个分支中随机选择一组cube
        for branch_id, cube_branch in enumerate(cube_branches):
            if len(cube_branch) > 0:
                # 随机选择一个cube 和 对应的idx
                selected_id = random.randint(0, len(cube_branch)-1)
                selected_cube = cube_branch[selected_id]
                selected_cube_idx = cube_branches_idxs[branch_id][selected_id]
                
                final_cubes+=selected_cube
                final_cubes_idxs+=selected_cube_idx
    if len(final_cubes) != 0:
        final_cubes = np.array(final_cubes)
    center = center_samples.reshape(-1, 8, 3)
    
    for center_idx, center_element in enumerate(center):
        # 将列表转换为numpy数组
        if len(final_cubes) != 0:
            
            if center_element is not None:

                final_cubes = np.concatenate([ final_cubes, center_element[None,...]], axis=0)
                final_cubes_idxs.append(center_samples_idxs[center_idx])
        else:
            final_cubes = center_element[None,...]
    cubes_first_frame = final_cubes
    cubes_first_frame_idxs = final_cubes_idxs
    cubes_all_frames = []
    for cube, idxs in zip(cubes_first_frame, cubes_first_frame_idxs):

        tracked_cube = track_a_cube(cube, idxs, joints)
        cubes_all_frames.append(tracked_cube[:,None,...])
    cubes_all_frames = np.concatenate(cubes_all_frames, axis=1)
    # cubes_all_frames = cubes_first_frame[None,...].repeat(joints.shape[0], axis=0)
    mp4path = os.path.join(save_path, savename)
    plot_3d_motion_with_cubes(mp4path, kinematic_chain , joints,cubes_all_frames,"test", figsize=(10, 10), radius=4, fps=24)

    print(f"final_cubes.shape: {final_cubes.shape}")  
    print(f"final_cubes_idxs: {final_cubes_idxs}")
    return 

if __name__ == "__main__":
    kinematic_chain = t2m_kinematic_chain
    id = 0
    joints = np.load(f'/home/yhc/momask-codes/dataset/HumanML3D/new_joints/{id:06d}.npy')

    save_path = f"./testCubeV2"
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    for center_mode in [1,2,3,4]:
        all_centers, all_branches, all_centers_idxs, all_branches_idxs = get_abstract_cube(joints, kinematic_chain, center_mode=center_mode)
        print(f"all_centers.shape: {len(all_centers)}")
        for _ in range(2):
            centering_id  = random.randint(0, len(all_centers) - 1)
            savename = f"test_{center_mode}_{centering_id}.mp4"
            # id indicates the centering strategy
            track_a_sample(all_branches[centering_id], 
                                all_branches_idxs[centering_id], 
                                all_centers[centering_id], 
                                all_centers_idxs[centering_id],
                                joints, savename)
