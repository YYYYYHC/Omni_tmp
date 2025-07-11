import numpy as np

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

def track_cubes(cubes, idxs, joints):
    '''
    given a set of cubes and their relative joints' idx, track the cubes based on the joints's motion
    
    return: a array of tracked cubes, shape (time, cube_num, 8, 3) following the joints' motion
    '''
    cubes_first_frame = cubes
    cubes_first_frame_idxs = idxs
    cubes_all_frames = []
    for cube, idxs in zip(cubes_first_frame, cubes_first_frame_idxs):

        tracked_cube = track_a_cube(cube, idxs, joints)
        cubes_all_frames.append(tracked_cube[:,None,...])
    cubes_all_frames = np.concatenate(cubes_all_frames, axis=1)
    return cubes_all_frames