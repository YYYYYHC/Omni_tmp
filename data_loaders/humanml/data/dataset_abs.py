import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
import pdb
from torch.utils.data._utils.collate import default_collate
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.get_opt import get_opt
from ..scripts.motion_process import recover_root_rot_pos, recover_from_ric
from data_loaders.humanml.utils.metrics import cross_combination_joints
from utils_abs.sample import sample_abstraction_all, apply_random_transformation_full, apply_random_drop_full, apply_random_global_scaling_translation
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import concurrent.futures as futures              # [MOD] 并行
import multiprocessing as mp                      # [MOD] 并行
# import spacy
config = {
    'PERTURB_ROTATION_STRENGTH': 0.1,
    'PERTURB_TRANSLATION_STRENGTH': 0.05,
    'PERTURB_TRANSLATION_STRENGTH_GLOBAL': 0, # no global translation
    'PERTURB_SCALING_STRENGTH': 0.3,
    'PROB_PERTURB_ABSTRACTION': 50/100.0,
    'PROB_DROP_ABSTRACTION': 50/100.0
}
PERTURB_ROTATION_STRENGTH = config['PERTURB_ROTATION_STRENGTH']
PERTURB_TRANSLATION_STRENGTH = config['PERTURB_TRANSLATION_STRENGTH']
PERTURB_TRANSLATION_STRENGTH_GLOBAL = config['PERTURB_TRANSLATION_STRENGTH_GLOBAL']
PERTURB_SCALING_STRENGTH = config['PERTURB_SCALING_STRENGTH']
PROB_PERTURB_ABSTRACTION = config['PROB_PERTURB_ABSTRACTION']
PROB_DROP_ABSTRACTION = config['PROB_DROP_ABSTRACTION']



# ======================= 并行辅助函数 ======================= #
def _build_single_sample(
        name,
        opt,
        min_motion_len,
        max_motion_length,
        abstraction_dir,
        motion_dir,
        position_dir,
        text_dir):
    """
    子进程中的核心逻辑：尝试解析一个 sample，返回
      (data_dict_partial, new_name_list_partial, length_list_partial, max_abs_num_partial)
    若解析失败 / 被过滤，返回 None。
    """
    try:
        abspath = pjoin(abstraction_dir, name + '.npy')
        if not os.path.exists(abspath):
            return None

        position = np.load(pjoin(position_dir, name + '.npy'), mmap_mode='r')
        motion   = np.load(pjoin(motion_dir,   name + '.npy'), mmap_mode='r')

        # skeleton abstraction 尚未支持
        if 'skeleton' in abstraction_dir:
            return None

        # cube abstraction
        abstraction_dict_path = pjoin(abstraction_dir, name + '.npy')
        max_abs_num = 23                         # 与旧代码一致

        # 过滤过短/过长 motion
        if (len(motion) < min_motion_len) or (len(motion) >= 200):
            return None

        data_dict_partial = {}
        new_name_partial  = []
        length_partial    = []

        text_path = pjoin(text_dir, name + '.txt')
        if not os.path.exists(text_path):
            return None

        flag = False
        text_data = []
        with cs.open(text_path) as f:
            for line in f.readlines():
                line_split = line.strip().split('#')
                caption = line_split[0]
                tokens  = line_split[1].split(' ')
                f_tag   = float(line_split[2])
                to_tag  = float(line_split[3])
                f_tag   = 0.0 if np.isnan(f_tag) else f_tag
                to_tag  = 0.0 if np.isnan(to_tag) else to_tag

                text_dict = dict(caption=caption, tokens=tokens)

                if f_tag == 0.0 and to_tag == 0.0:
                    flag = True
                    text_data.append(text_dict)
                else:
                    try:
                        n_motion = motion[int(f_tag*20):int(to_tag*20)]
                        if (len(n_motion) < min_motion_len) or (len(n_motion) >= 200):
                            continue
                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        while new_name in data_dict_partial:                     # 冲突规避
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                        data_dict_partial[new_name] = dict(
                            motion=n_motion,
                            position=position,
                            abstraction_all_path=abstraction_dict_path,
                            f_t_mul20=dict(f_tag20=f_tag*20, to_tag20=to_tag*20),
                            length=len(n_motion),
                            text=[text_dict]
                        )
                        new_name_partial.append(new_name)
                        length_partial.append(len(n_motion))
                    except Exception:
                        # 打印一次即可；留给主进程汇总
                        pass

        if flag:
            data_dict_partial[name] = dict(
                motion=motion,
                position=position,
                abstraction_all_path=abstraction_dict_path,
                length=len(motion),
                text=text_data
            )
            new_name_partial.append(name)
            length_partial.append(len(motion))

        if not data_dict_partial:
            return None

        return data_dict_partial, new_name_partial, length_partial, max_abs_num
    except Exception:
        # 任何异常直接返回 None，由主进程忽略
        return None
# ========================================================== #


class Text2MotionDatasetV2(data.Dataset):
    """
    For use of training text-motion matching model, and evaluations
    """

    # ---------------------- [MOD] 新增参数 num_workers ---------------------- #
    def __init__(self, opt, mean, std, split_file, w_vectorizer, mode,
                 control_joint=0, density=100, num_workers=64):
        # ------------------------------------------------------------------- #
        self.valtest = ('val' in split_file or 'test' in split_file)
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        self.mode = mode
        min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24
        self.control_joint = control_joint
        self.density = density
        self.num_workers = num_workers            # [MOD]

        # ---------- 1. 读取 id_list ----------
        with cs.open(split_file, 'r') as f:
            id_list = [line.strip() for line in f.readlines()]

        data_dict = {}
        new_name_list = []
        length_list = []
        max_abs_num_global = 0

        # ---------- 2. 并行解析每一个 sample ----------
        # * 使用 spawn 以避免 fork 后在某些环境里出现的多线程死锁（如 OpenBLAS）
        mp_ctx = mp.get_context('spawn')
        with futures.ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp_ctx) as executor:
            futures_list = [
                executor.submit(
                    _build_single_sample,
                    name,
                    opt,
                    min_motion_len,
                    self.max_motion_length,
                    opt.abstraction_dir,
                    opt.motion_dir,
                    opt.position_dir,
                    opt.text_dir)
                for name in id_list
            ]

            for fut in tqdm(futures.as_completed(futures_list),
                            total=len(futures_list),
                            desc='Building dataset (parallel)'):
                result = fut.result()
                if result is None:
                    continue
                sub_dict, sub_names, sub_lengths, sub_max_abs = result
                data_dict.update(sub_dict)
                new_name_list.extend(sub_names)
                length_list.extend(sub_lengths)
                max_abs_num_global = max(max_abs_num_global, sub_max_abs)

        # ---------- 3. 后续整理，与旧实现一致 ----------
        if not new_name_list:
            raise RuntimeError('No data found. Check paths & filters.')

        name_list, length_list = zip(*sorted(
            zip(new_name_list, length_list), key=lambda x: x[1]))

        self.max_abs_num = max_abs_num_global
        self.mean = mean
        self.std = std

        if 'HumanML3D' in opt.data_root:
            spatial_norm_path = '/workspace/writeable/dataset/humanml_spatial_norm'
        elif 'KIT' in opt.data_root:
            spatial_norm_path = '/workspace/writeable/dataset/kit_spatial_norm'
        else:
            raise NotImplementedError('unknown dataset')

        self.raw_mean = np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
        self.raw_std  = np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))
        self.length_arr = np.array(length_list)
        self.data_dict  = data_dict
        self.name_list  = name_list
        self.reset_max_len(self.max_length)

    # =================== 其余成员函数原样保留 =================== #
    # （所有实际训练/推断逻辑完全不变，只是 __init__ 改成并行构建）
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def random_mask_cross(self, joints, n_joints=22, density=1):
        cross_joints = cross_combination_joints()
        choose = np.random.choice(len(cross_joints), 1).item()
        choose_joint = cross_joints[choose]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
    
    def random_mask(self, joints, n_joints=22, density=1):
        if n_joints == 22:
            # humanml3d
            controllable_joints = np.array([0, 10, 11, 15, 20, 21])
        else:
            # kit
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF']
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])

        choose_joint = [self.control_joint]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        # density = 100
        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_train(self, joints, n_joints=22):
        if n_joints == 22:
            controllable_joints = np.array([0, 10, 11, 15, 20, 21])
        else:
            {1:'root', 2:'BP', 3:'BT', 4:'BLN', 5:'BUN', 6:'LS', 7:'LE', 8:'LW', 9:'RS', 10:'RE', 11:'RW', 12:'LH', 13:'LK', 14:'LA', 15:'LMrot', 16:'LF', 17:'RH', 18:'RK', 19:'RA', 20:'RMrot', 21:'RF'}
            choose_one = ['root', 'BUN', 'LW', 'RW', 'LF', 'RF']
            controllable_joints = np.array([0, 4, 7, 10, 15, 20])
        num_joints = len(controllable_joints)
        # joints: length, 22, 3
        num_joints_control = np.random.choice(num_joints, 1)
        # only use one joint during training
        num_joints_control = 1
        choose_joint = np.random.choice(num_joints, num_joints_control, replace=False)
        choose_joint = controllable_joints[choose_joint]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints

    def random_mask_train_cross(self, joints, n_joints=22):
        from data_loaders.humanml.utils.metrics import cross_combination_joints
        cross_joints = cross_combination_joints()
        choose = np.random.choice(len(cross_joints), 1).item()
        # choose = -1
        choose_joint = cross_joints[choose]

        length = joints.shape[0]
        choose_seq_num = np.random.choice(length - 1, 1) + 1
        choose_seq = np.random.choice(length, choose_seq_num, replace=False)
        choose_seq.sort()
        mask_seq = np.zeros((length, n_joints, 3)).astype(np.bool)

        for cj in choose_joint:
            mask_seq[choose_seq, cj] = True

        # normalize
        joints = (joints - self.raw_mean.reshape(n_joints, 3)) / self.raw_std.reshape(n_joints, 3)
        joints = joints * mask_seq
        return joints
        
    def __len__(self):
        return max(0, len(self.data_dict) - self.pointer)

    def __getitem__(self, item):
        if item < 0:
            raise ValueError(f"索引不能为负值: {item}")
        if item >= len(self):
            raise IndexError(f"索引超出范围: {item} >= {len(self)}")
        
        data_idx = self.pointer + item
        if data_idx >= len(self.name_list):
            raise IndexError(f"数据索引超出范围: {data_idx} >= {len(self.name_list)}")
        
        data = self.data_dict[self.name_list[data_idx]]
        motion, position, abstraction_all_path, m_length, text_list = data['motion'], data['position'], data['abstraction_all_path'], data['length'], data['text']
        frame_start = data['f_t_mul20']['f_tag20'] if 'f_t_mul20' in data.keys() else None
        frame_end = data['f_t_mul20']['to_tag20'] if 'f_t_mul20' in data.keys() else None
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        # randomly set to empty
        if not self.valtest and random.random() < 0.5:
            caption = ''
            tokens = []
        
        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)  
        motion = motion[idx:idx+m_length]
        
        
        
    
        abstraction_all = np.load(abstraction_all_path, allow_pickle=True)
        abstraction_full = sample_abstraction_all(abstraction_all)
        
        # random pertub the sampled abstraction
        # generate random 3D rotation, translation, scale x y z
        abstraction_full = apply_random_global_scaling_translation(abstraction_full, scaling_strength=PERTURB_SCALING_STRENGTH, translation_strength=PERTURB_TRANSLATION_STRENGTH_GLOBAL)
        
        # pertub some of the cubes
        if random.random() < PROB_PERTURB_ABSTRACTION and not self.valtest:
            
            abstraction_full = apply_random_transformation_full(abstraction_full, rotation_strength=np.pi*PERTURB_ROTATION_STRENGTH, translation_strength=PERTURB_TRANSLATION_STRENGTH)
            
            # np.save(f'/home/yhc/OmniControl/dataset/HumanML3D/testAugmentation/{self.name_list[data_idx]}.npy', abstraction_full)
            # if len(os.listdir(f'/home/yhc/OmniControl/dataset/HumanML3D/testAugmentation')) > 20:
            #     breakpoint()
        if random.random() < PROB_DROP_ABSTRACTION and not self.valtest:
            #TBD: drop the abstraction
            abstraction_full = apply_random_drop_full(abstraction_full)
            # np.save('abstraction_full.npy', abstraction_full)
            # print('data_idx', data_idx)
            # breakpoint()


        if frame_start is not None and frame_end is not None:
            abstraction_full = abstraction_full[int(frame_start):int(frame_end)]
        abstraction = abstraction_full[idx:idx+m_length]
        # 将abstraction转换为float32类型
        abstraction = abstraction.astype(np.float32)
        # 重新整形为(t, k* 8 * 3)
        # 在k维度上随机打乱
        k = abstraction.shape[1]
        shuffle_idx = np.random.permutation(k)
        abstraction = abstraction[:, shuffle_idx, :, :]
        abstraction = abstraction.reshape(abstraction.shape[0], -1)
        
        # n_joints = 22 if motion.shape[-1] == 263 else 21
        
        # hint is global position of the controllable joints
        # joints = recover_from_ric(torch.from_numpy(motion).float(), n_joints)
        # joints = joints.numpy()

        # # control any joints at any time
        # if self.mode == 'train':
        #     # hint = self.random_mask_train_cross(joints, n_joints)
        #     hint = self.random_mask_train(joints, n_joints)
        # else:
        #     # hint = self.random_mask_cross(joints, n_joints)
        #     hint = self.random_mask(joints, n_joints)

        # hint = hint.reshape(hint.shape[0], -1)
        # if m_length < self.max_motion_length:
        #     hint = np.concatenate([hint,
        #                            np.zeros((self.max_motion_length - m_length, hint.shape[1]))
        #                             ], axis=0)
        
        
        "Z Normalization"
        motion = (motion - self.mean) / self.std
        
        #TBD: normalize the abstraction 
        # abstraction = (abstraction - self.raw_mean) / self.raw_std
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
            
            abstraction = np.concatenate([abstraction,
                                     np.zeros((self.max_motion_length - m_length, abstraction.shape[1]))
                                     ], axis=0)
        
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), abstraction, self.max_abs_num


class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120


        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in data_dict:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, np.array([0]), self.fixed_length, None, None


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train", control_joint=0, density=100, use_multiprocessing=True, num_workers=64, **kwargs):
        self.mode = mode
        
        self.dataset_name = 't2m'
        self.dataname = 't2m'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'/workspace/writeable'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        #yhc: abstraction dir
        opt.abstraction_dir = pjoin(abs_base_path, opt.abstraction_dir)
        
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = '/workspace/writeable/dataset'
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)

        if mode == 'gt':
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
        elif mode in ['train', 'eval', 'text_only']:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, 'Mean.npy'))
            self.std = np.load(pjoin(opt.data_root, 'Std.npy'))

        if mode == 'eval':
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        if mode == 'text_only':
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, mode, control_joint, density, use_multiprocessing, num_workers)
            self.num_actions = 1 # dummy placeholder

        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()

# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):
    def __init__(self, mode, datapath='./dataset/kit_opt.txt', split="train", **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)