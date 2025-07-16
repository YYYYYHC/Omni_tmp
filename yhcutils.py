import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 必须导入
from matplotlib.animation import FuncAnimation

def sample_joints(joints, kinematic_chain, sample_dis=0.01):
    '''
    Given a joints sequence and a kinematic chain, we sample a sequence of pointcloud
    
    joints: (N, 22, 3) time sequence
    kinematic_chain: kinematic of SMPL
    point_num: number of points after sampling
    
    return: pointcloud and kinematic chain after sampling
    '''
    # in the first frame, we get the number of points to be sampled for each line
    joints_f0 = joints[0]
    kinematic_chain_points_num = []
    for kinematic_chain_block in kinematic_chain:
        joints_f0_kcb = joints_f0[kinematic_chain_block]
        #get lines length
        lines_length = np.sqrt(np.sum((joints_f0_kcb[0:-1] - joints_f0_kcb[1:])**2, axis= -1))
        #get number of points per line
        lines_point_num = list(np.int_(lines_length // sample_dis))
        kinematic_chain_points_num.append(lines_point_num)
    #sampling all frames
    all_new_points = []
    new_kinemetic_chain = []
    current_id_pointer = 0
    for kinematic_chain_block, kinematic_chain_block_points_num in zip(kinematic_chain, kinematic_chain_points_num):
        joints_kcb = joints[:, kinematic_chain_block, :]
        block_new_points = []
        block_new_kinemetic_chain = []
        block_point_num = 0
        for i, pn in enumerate(kinematic_chain_block_points_num):
            points_a = joints_kcb[:, i,:]            
            points_b = joints_kcb[:, i+1, :]
            new_points = points_a[:, None] + (points_b[:, None] - points_a[:, None]) * np.linspace(0, 1, pn, endpoint=False)[None, :, None]
            block_new_points.append(new_points)
            block_point_num += pn
            block_new_kinemetic_chain.append([i for i in range(current_id_pointer, current_id_pointer + pn)])
            current_id_pointer += pn
        block_new_points = np.concatenate(block_new_points, axis=1)
        all_new_points.append(block_new_points)
        new_kinemetic_chain.append(block_new_kinemetic_chain)
    all_new_points = np.concatenate(all_new_points, axis=1)
    return (all_new_points, new_kinemetic_chain)

def pcaCube(X):
    '''
    compute pca bbox given a set of points
    '''
    # 2. 计算 PCA
    # 2.1 去中心化
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X

    # 2.2 协方差矩阵 & 特征分解 (也可使用np.linalg.svd)
    cov_mat = np.cov(X_centered.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # 特征值从大到小排序，特征向量同步排序
    order = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[order]
    eig_vecs = eig_vecs[:, order]

    # 3. 计算包围盒(OBB)
    # 3.1 将数据投影到主轴空间
    X_pca = X_centered @ eig_vecs  # (n,3)

    # 3.2 在投影空间找 min/max
    min_bounds = np.min(X_pca, axis=0)
    max_bounds = np.max(X_pca, axis=0)
    
    # 3.3 得到包围盒在投影空间的 8 个顶点
    # 在3D空间中，每个顶点可以由 min 和 max 的组合确定
    # 例如 (x_min, y_min, z_min), (x_min, y_min, z_max), ...
    corners_pca = np.array([
        [min_bounds[0], min_bounds[1], min_bounds[2]],
        [min_bounds[0], min_bounds[1], max_bounds[2]],
        [min_bounds[0], max_bounds[1], min_bounds[2]],
        [min_bounds[0], max_bounds[1], max_bounds[2]],
        [max_bounds[0], min_bounds[1], min_bounds[2]],
        [max_bounds[0], min_bounds[1], max_bounds[2]],
        [max_bounds[0], max_bounds[1], min_bounds[2]],
        [max_bounds[0], max_bounds[1], max_bounds[2]],
    ])

    # 3.4 将包围盒顶点从 PCA 空间转换回原始坐标系
    # 注意：X_centered = X - mean_X,  X_pca = X_centered @ eig_vecs
    # 则反变换 corners_original = corners_pca @ eig_vecs^T + mean_X
    cube = corners_pca @ eig_vecs.T + mean_X
    return cube

def pcaCubeBatch(X, min_length=0.0):
    """
    X: shape (b, n, 3), 其中 b 为批量大小, n 为每批的点数
    min_length: float, 包围盒每个维度的最小长度
    return: shape (b, 8, 3), 每个批次对应的 PCA OBB 的 8 个顶点
    """
    b, n, _ = X.shape
    # 用来存储每个批次计算的 8 个角点
    all_corners = np.zeros((b, 8, 3), dtype=X.dtype)
    
    for i in range(b):
        # 取第 i 个批次的数据，形状为 (n, 3)
        pts = X[i]
        
        # 1. 去中心化
        mean_pts = np.mean(pts, axis=0)
        pts_centered = pts - mean_pts
        
        # 2. 计算协方差矩阵 & 特征分解
        cov_mat = np.cov(pts_centered.T)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        # 根据特征值大小对特征向量排序
        order = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[order]
        eig_vecs = eig_vecs[:, order]
        
        # 3. 求包围盒
        # 3.1 将数据投影到 PCA 主轴空间 (n,3)
        pts_pca = pts_centered @ eig_vecs
        
        # 3.2 在投影空间找 min/max
        min_bounds = np.min(pts_pca, axis=0)
        max_bounds = np.max(pts_pca, axis=0)
        
        # 3.3 确保每个维度至少有 min_length 长度
        lengths = max_bounds - min_bounds
        for j in range(3):
            if lengths[j] < min_length:
                center = (max_bounds[j] + min_bounds[j]) / 2
                half_length = min_length / 2
                min_bounds[j] = center - half_length
                max_bounds[j] = center + half_length
        
        # 3.4 构造 8 个顶点 (在 PCA 坐标系中)
        corners_pca = np.array([
            [min_bounds[0], min_bounds[1], min_bounds[2]],
            [min_bounds[0], min_bounds[1], max_bounds[2]],
            [min_bounds[0], max_bounds[1], min_bounds[2]],
            [min_bounds[0], max_bounds[1], max_bounds[2]],
            [max_bounds[0], min_bounds[1], min_bounds[2]],
            [max_bounds[0], min_bounds[1], max_bounds[2]],
            [max_bounds[0], max_bounds[1], min_bounds[2]],
            [max_bounds[0], max_bounds[1], max_bounds[2]],
        ])
        
        # 3.5 将 corners_pca 变换回原始坐标系
        corners_world = corners_pca @ eig_vecs.T + mean_pts
        
        # 存储到结果中
        all_corners[i] = corners_world
        
    return all_corners

def get_template_cube(R=np.eye(3), T=np.zeros(3), S=np.ones(3)):
    # get the teamplace cube and apply rotation, translation, and scaling in the canonical space
    min_bounds = np.array([-1, -1, -1])
    max_bounds = np.array([1, 1, 1])

    # 3.3 得到包围盒在投影空间的 8 个顶点
    # 在3D空间中，每个顶点可以由 min 和 max 的组合确定
    # 例如 (x_min, y_min, z_min), (x_min, y_min, z_max), ...
    tamplateCube = np.array([
        [min_bounds[0], min_bounds[1], min_bounds[2]],
        [min_bounds[0], min_bounds[1], max_bounds[2]],
        [min_bounds[0], max_bounds[1], min_bounds[2]],
        [min_bounds[0], max_bounds[1], max_bounds[2]],
        [max_bounds[0], min_bounds[1], min_bounds[2]],
        [max_bounds[0], min_bounds[1], max_bounds[2]],
        [max_bounds[0], max_bounds[1], min_bounds[2]],
        [max_bounds[0], max_bounds[1], max_bounds[2]],
    ])
    # apply rotation, translation, and scaling
    tamplateCube = tamplateCube @ R.T
    tamplateCube = tamplateCube @ np.diag(S)
    tamplateCube = tamplateCube + T
    return tamplateCube

def pcaCube(X, mins = 0.05):
    '''
    compute pca bbox given a set of points
    X: shape (n, 3)
    '''
    # 2. 计算 PCA
    # 2.1 去中心化
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X

    # 2.2 协方差矩阵 & 特征分解 (也可使用np.linalg.svd)
    cov_mat = np.cov(X_centered.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # 特征值从大到小排序，特征向量同步排序
    order = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[order]
    eig_vecs = eig_vecs[:, order]

    # 3. 计算包围盒(OBB)
    # 3.1 将数据投影到主轴空间
    X_pca = X_centered @ eig_vecs  # (n,3)

    # 3.2 在投影空间找 min/max
    min_bounds = np.min(X_pca, axis=0)
    max_bounds = np.max(X_pca, axis=0)
    lengths = max_bounds - min_bounds
    min_bounds = np.where(lengths < mins, -mins/2, min_bounds)
    max_bounds = np.where(lengths < mins, mins/2, max_bounds)
    # 3.3 得到包围盒在投影空间的 8 个顶点
    # 在3D空间中，每个顶点可以由 min 和 max 的组合确定
    # 例如 (x_min, y_min, z_min), (x_min, y_min, z_max), ...
    corners_pca = np.array([
        [min_bounds[0], min_bounds[1], min_bounds[2]],
        [min_bounds[0], min_bounds[1], max_bounds[2]],
        [min_bounds[0], max_bounds[1], min_bounds[2]],
        [min_bounds[0], max_bounds[1], max_bounds[2]],
        [max_bounds[0], min_bounds[1], min_bounds[2]],
        [max_bounds[0], min_bounds[1], max_bounds[2]],
        [max_bounds[0], max_bounds[1], min_bounds[2]],
        [max_bounds[0], max_bounds[1], max_bounds[2]],
    ])

    # 3.4 将包围盒顶点从 PCA 空间转换回原始坐标系
    # 注意：X_centered = X - mean_X,  X_pca = X_centered @ eig_vecs
    # 则反变换 corners_original = corners_pca @ eig_vecs^T + mean_X
    cube = corners_pca @ eig_vecs.T + mean_X
    return cube
def line2cube(p1, p2, templateCube):
    '''
    Given two 3D points, return the cube
    p1: shape(t, 3)
    p2: shape(t, 3)
    '''
    pup = np.where(p1[:, 2:] > p2[:, 2:], p1, p2)
    plw = np.where(p1[:, 2:] > p2[:, 2:], p2, p1)
    # Compute the transformation matrix which 
    # transform the line (p1, p2) to ((0,0,1), (0,0,-1))
    # 计算从 (pup, plw) 到 ((0,0,1), (0,0,-1)) 的变换矩阵
    # 1. 计算线段的方向向量并归一化
    dir_vec = pup - plw
    dir_vec = dir_vec / np.linalg.norm(dir_vec, axis=1, keepdims=True)
    
    # 2. 计算旋转矩阵 R，将 dir_vec 旋转到 z 轴 (0,0,1)
    z_axis = np.array([[0, 0, 1]])
    rotation_axis = np.cross(dir_vec, z_axis)
    # 如果方向向量已经与z轴平行,不需要旋转
    R = np.eye(3)[None, :, :].repeat(dir_vec.shape[0], axis=0)
    parallel_mask = np.linalg.norm(rotation_axis, axis=1) < 1e-8
    
    if np.any(parallel_mask):
        R[parallel_mask] = np.where(dir_vec[parallel_mask, 2:3] < 0, -R[parallel_mask], R[parallel_mask])
    
    # 对于不平行的情况，计算旋转矩阵
    non_parallel_mask = ~parallel_mask
    if np.any(non_parallel_mask):
        rotation_axis[non_parallel_mask] = rotation_axis[non_parallel_mask] / np.linalg.norm(rotation_axis[non_parallel_mask], axis=1, keepdims=True)
        cos_theta = np.einsum('ij,ij->i', dir_vec[non_parallel_mask], z_axis)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        # Rodrigues旋转公式
        K = np.zeros((dir_vec.shape[0], 3, 3))
        K[non_parallel_mask, 0, 1] = -rotation_axis[non_parallel_mask, 2]
        K[non_parallel_mask, 0, 2] = rotation_axis[non_parallel_mask, 1]
        K[non_parallel_mask, 1, 0] = rotation_axis[non_parallel_mask, 2]
        K[non_parallel_mask, 1, 2] = -rotation_axis[non_parallel_mask, 0]
        K[non_parallel_mask, 2, 0] = -rotation_axis[non_parallel_mask, 1]
        K[non_parallel_mask, 2, 1] = rotation_axis[non_parallel_mask, 0]
        R[non_parallel_mask] = np.eye(3) + sin_theta[:, None, None] * K[non_parallel_mask] + (1 - cos_theta)[:, None, None] * (K[non_parallel_mask] @ K[non_parallel_mask])

    # 使用广播机制逐元素相乘
    pup_rotated = np.einsum('ijk,ik->ij', R, pup)
    plw_rotated = np.einsum('ijk,ik->ij', R, plw)
    
    assert np.linalg.norm(pup_rotated[:, :2] - plw_rotated[:, :2])< 1e-6, "the line should be parallel to the z axis"
    # 4. 计算平移向量
    T = - (pup_rotated + plw_rotated) / 2
    # 计算缩放因子S，假设基于pup_rotated和plw_rotated的z轴差异
    z_diff = np.abs(pup_rotated[:, 2] - plw_rotated[:, 2])
    S = np.stack([np.ones_like(z_diff), np.ones_like(z_diff), 2 / z_diff], axis=1)
    
    # 5. 将模板立方体应用逆变换到线段空间
    # 逆变换顺序：先缩放，再平移，最后旋转
    cube = np.einsum('ik,ijk->ijk', 1/S, templateCube)

    cube = cube - T[:, None, :]  # 平移变换
    
    cube = np.einsum('ikl, ijk->ijl',R, cube)
    return cube
    
    

def animate_cubes(cube_data, points, sampled_idx, interval=500):
    """
    用于可视化 (t, k, 8, 3) 的 OBB 数据的动画。
    - cube_data: shape = (t, k, 8, 3), 
        t为时间步数, k为每个时间步的OBB数, 
        8为每个OBB的顶点, 3为(x,y,z)
    - interval: 每帧之间的间隔(毫秒)
    Returns:
        ani: FuncAnimation 对象
    """
    EDGES = [
    (0, 1), (0, 2), (0, 4),
    (3, 1), (3, 2), (3, 7),
    (5, 1), (5, 4), (5, 7),
    (6, 2), (6, 4), (6, 7)
    ]
    t, k, _, _ = cube_data.shape

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title("Animation of OBBs Over Time")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 1) 初始化存放所有立方体的绘图对象
    # 我们需要：每个立方体对应一个 scatter (8点) + 12 条线段
    # 为了方便更新，把这些对象都保存在列表里
    scatters = []
    lines = []
    
    x_min, x_max = points[:,:,0].min(), points[:,:,0].max()
    y_min, y_max = points[:,:,1].min(), points[:,:,1].max()
    z_min, z_max = points[:,:,2].min(), points[:,:,2].max()
    # 确保坐标轴比例一致
    max_range = max(x_max-x_min, y_max-y_min, z_max-z_min)
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    mid_z = (z_max + z_min) / 2
    
    # 创建两个点云散点图对象：一个用于普通点，一个用于采样点
    point_cloud = ax.scatter([], [], [], color='gray', alpha=0.3, s=10, label='原始点云')
    sampled_points = ax.scatter([], [], [], color='black', alpha=1.0, s=60, label='采样点')
    
    for _ in range(k):
        # 先创建 scatter, 使用空数据占位，后续更新
        sc = ax.scatter([], [], [], s=50, marker='o')
        scatters.append(sc)

        # 再创建 12 条边对应的 line 对象
        cube_lines = []
        for __ in range(len(EDGES)):
            ln, = ax.plot([], [], [], linewidth=1.0) 
            cube_lines.append(ln)
        lines.append(cube_lines)

    # 2) 让坐标轴范围大一点，以防止立方体跑出视野
    # 这里只是简单设置一个较大的范围，也可以根据数据做自适应
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    ax.view_init(elev=90, azim=-90)
    # 3) 定义 init_func：初始化时调用一次
    def init():
        # 这里只要返回要刷新的 artists 即可
        artists = []
        for sc in scatters:
            artists.append(sc)
        for cube_lines in lines:
            for ln in cube_lines:
                artists.append(ln)
        return artists

    # 4) 定义 update(frame)：更新第 frame 帧数据
    def update(frame):
        points_frame = points[frame]
        
        # 更新普通点云
        mask = np.ones(len(points_frame), dtype=bool)
        mask[sampled_idx] = False
        normal_points = points_frame[mask]
        point_cloud._offsets3d = (normal_points[:,0], normal_points[:,1], normal_points[:,2])
        
        # 更新采样点
        sampled_points_data = points_frame[sampled_idx]
        sampled_points._offsets3d = (sampled_points_data[:,0], sampled_points_data[:,1], sampled_points_data[:,2])
            
        # cube_data[frame] 的形状是 (k, 8, 3)
        # 里面是本帧下 k 个立方体的 8 点坐标
        current = cube_data[frame]
        
        # 逐个立方体更新可视化
        for i in range(k):
            corners = current[i]  # shape = (8, 3)

            # (i) 更新 scatter
            # Matplotlib 3D 散点可以这样更新
            scatters[i]._offsets3d = (
                corners[:,0], 
                corners[:,1], 
                corners[:,2]
            )

            # (ii) 更新 12 条 line
            for e_idx, edge in enumerate(EDGES):
                p1 = corners[edge[0]]
                p2 = corners[edge[1]]
                # 画线
                lines[i][e_idx].set_data_3d(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]]
                )

        # 返回所有更新的 artists
        artists = []
        for sc in scatters:
            artists.append(sc)
        for cube_lines in lines:
            for ln in cube_lines:
                artists.append(ln)
        return artists

    # 5) 创建动画
    ani = FuncAnimation(
        fig, 
        update, 
        frames=t,       # 帧数
        init_func=init, 
        interval=interval, 
        blit=True      # 3D 动画通常只能 blit=False
    )

    return ani

def visualize_pcaCube(X, cube):
        # 4. 可视化
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 4.1 绘制原始点云
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.3, label='Point Cloud')

    # 4.2 绘制包围盒
    # 需要将8个顶点按照立方体的边进行连接
    # 立方体的边的连接可以按照下面的索引规则（每条边两个顶点）:
    edges = [
        (0, 1), (0, 2), (0, 4),
        (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7),
        (6, 2), (6, 4), (6, 7)
    ]

    # 绘制包围盒的 8 个顶点
    ax.scatter(cube[:, 0],
            cube[:, 1],
            cube[:, 2],
            s=50, marker='o', label='OBB corners')

    # 绘制包围盒的 12 条线段
    for edge in edges:
        p1 = cube[edge[0]]
        p2 = cube[edge[1]]
        xs = [p1[0], p2[0]]
        ys = [p1[1], p2[1]]
        zs = [p1[2], p2[2]]
        ax.plot(xs, ys, zs, linewidth=1.0)
    return
# if __name__ == "__main__":
#     n = 200
#     # 这里构造一个稍微倾斜的点云，用于更好地演示 PCA OBB
#     X = np.dot(np.random.rand(n, 3) - 0.5, 
#            [[1.0, 0.2, 0.0],
#             [0.0, 0.7, 0.3],
#             [0.0, 0.0, 1.0]]) * 10

#     cube = pcaCube(X)
#     visualize_pcaCube(X, cube)
