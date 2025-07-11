# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# plt.rcParams['animation.ffmpeg_path'] = '/shared/centos7/ffmpeg/20190305/bin/ffmpeg'
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=[], hint=None, hintOnly=False, lineOnly=False):
    matplotlib.use('Agg')
    EDGES = [
    (0, 1), (0, 2), (0, 4),
    (3, 1), (3, 2), (3, 7),
    (5, 1), (5, 4), (5, 7),
    (6, 2), (6, 4), (6, 7)
    ]
    
    title = '\n'.join(wrap(title, 20))

    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)


    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    # preparation related to specific datasets
    if dataset == 'kit':
        data *= 0.003  # scale for visualization
        if hint is not None:
            if len(hint.shape)==3:
                mask = hint.sum(-1) != 0
                hint = hint[mask]
            hint *= 0.003
    elif dataset == 'humanml':
        data *= 1.3  # scale for visualization
        if hint is not None:
            if len(hint.shape)==3:
                mask = hint.sum(-1) != 0
                hint = hint[mask]
            if len(hint.shape) == 4 and hint.shape[2] == 8:
                k = hint.shape[1]
                
            hint *= 1.3


    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue

    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    if hint is not None:
        hint[..., 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]


    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])

        if hint is not None:
          
            if len(hint.shape)==2:
                ax.scatter(hint[..., 0] - trajec[index, 0], hint[..., 1], hint[..., 2] - trajec[index, 1], color="#80B79A")
            elif len(hint.shape) == 4 and hint.shape[2] == 2:
                for i in range(hint.shape[1]):
                    line = hint[index,i,:,:]
                    ax.plot3D([line[0,0]-trajec[index, 0], line[1,0]-trajec[index, 0]], [line[0,1], line[1,1]], [line[0,2]-trajec[index, 1], line[1,2]-trajec[index, 1]], color="#80B79A")
            elif len(hint.shape) == 4 and hint.shape[2] == 8:
                current = hint[index]
                #plot cubes
                for i in range(k):
                    corners = current[i]  # shape = (8, 3)

                    # 使用 ax.plot3D 更新 scatter
                    if not lineOnly:
                        ax.plot3D(corners[:,0]-trajec[index, 0], corners[:,1], corners[:,2]-trajec[index, 1], 'o', markersize=5)

                    # 使用 ax.plot3D 更新 12 条 line
                    for e_idx, edge in enumerate(EDGES):
                        p1 = corners[edge[0]]
                        p2 = corners[edge[1]]
                        # 画线
                        if not lineOnly:
                            ax.plot3D(
                                [p1[0]-trajec[index, 0], p2[0]-trajec[index, 0]],
                                [p1[1], p2[1]],
                                [p1[2]-trajec[index, 1], p2[2]-trajec[index, 1 ]],
                                linewidth=1.0
                            )

                
        used_colors = colors_blue if index in gt_frames else colors
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            if not hintOnly:
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                          color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()
