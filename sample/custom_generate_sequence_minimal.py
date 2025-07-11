import os
import numpy as np
import torch
from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from model.cfg_sampler import ClassifierFreeSampleModel
from utils import dist_util
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    fps = 20
    n_frames = 196
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                f'samples_{name}_{niter}_seed{args.seed}')
    args.batch_size = args.num_samples

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, None)
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    # 直接读取hint
    hint = np.load(args.hint_path)  # 你需要在参数中指定hint_path
    center = np.mean(hint[0].reshape(-1,3), axis=0)
    extent = np.max(hint[0].reshape(-1,3), axis=0) - np.min(hint[0].reshape(-1,3), axis=0)
    target_center = np.array([0, extent[1]/2, 0])
    translation = target_center - center
    scale = 1.5 / extent[1]
    hint = hint + translation
    hint = hint * scale
    n_frames = hint.shape[0]
    hint = np.repeat(hint[None,:], args.batch_size, axis=0)
    hint = hint.reshape(hint.shape[0], hint.shape[1], -1)
    current_vec_len = hint.shape[2]
    hint = np.pad(hint, ((0, 0), (0, 0), (0, 552-current_vec_len)), mode='constant', constant_values=0)
    model_kwargs = {'y': {}}
    model_kwargs['y']['hint'] = torch.tensor(hint, device=dist_util.dev(), dtype=torch.float)
    model_kwargs['y']['text'] = [''] * args.batch_size

    sample_fn = diffusion.p_sample_loop
    sample = sample_fn(
        model,
        (args.batch_size, model.njoints, model.nfeats, n_frames),
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    sample = sample[:, :263]
    sample = model.rot2xyz(x=sample, mask=None, pose_rep='xyz', glob=True, translation=True,
                           jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                           get_rotations_back=False)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)

    # 可视化输出mp4
    from data_loaders.humanml.utils import paramUtil
    skeleton = paramUtil.t2m_kinematic_chain
    for i in range(args.batch_size):
        motion = sample[i].cpu().numpy().transpose(1, 0, 2)  # (seq, joints, 3)
        caption = model_kwargs['y']['text'][i]
        save_path = os.path.join(out_path, f'sample{i:02d}.mp4')
        plot_3d_motion(save_path, skeleton, motion, dataset='humanml', title=caption, fps=fps, hint=hint[i])
        plot_3d_motion(save_path.replace('.mp4', '_hint.mp4'), skeleton, motion, dataset='humanml', title=caption, fps=fps, hint=hint[i], hintOnly=True)
        plot_3d_motion(save_path.replace('.mp4', '_line.mp4'), skeleton, motion, dataset='humanml', title=caption, fps=fps, hint=hint[i], lineOnly=True)
    print(f'[Done] Results are at [{os.path.abspath(out_path)}]')

if __name__ == "__main__":
    main() 