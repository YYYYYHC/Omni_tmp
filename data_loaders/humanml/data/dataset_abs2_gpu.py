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
def collate_fn(batch):
    if batch[0][-1] is None:
        batch = [b[:-1] for b in batch]
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)




# ─────────────────────────────────────────────────────────────────────────────
class Text2MotionDatasetV2(data.Dataset):
    r"""
    GPU-friendly 版本：
    1. 读盘后立即 to(device)
    2. 所有随机裁剪／几何扰动／mask 都用 torch
    """
    def __init__(self, opt, mean, std, split_file, w_vectorizer,
                 mode, control_joint=0, density=100, device="cuda") -> None:
        super().__init__()
        self.valtest      = ('val' in split_file or 'test' in split_file)
        self.opt          = opt
        self.max_text_len = opt.max_text_len
        self.max_motion_length = opt.max_motion_length
        self.mode         = mode
        self.control_joint = control_joint
        self.density      = density
        self.device       = torch.device(device)

        # 1. 预加载 meta
        self.mean = torch.as_tensor(mean, device=self.device, dtype=torch.float32)
        self.std  = torch.as_tensor(std,  device=self.device, dtype=torch.float32)

        # raw mean/std 用于 joints mask
        if 'HumanML3D' in opt.data_root:
            spatial_norm_path = '/workspace/writeable/dataset/humanml_spatial_norm'
        elif 'KIT' in opt.data_root:
            spatial_norm_path = '/workspace/writeable/dataset/kit_spatial_norm'
        else:
            raise NotImplementedError('unknown dataset')
        self.raw_mean = torch.from_numpy(
            np.load(pjoin(spatial_norm_path, 'Mean_raw.npy'))
        ).to(self.device)
        self.raw_std  = torch.from_numpy(
            np.load(pjoin(spatial_norm_path, 'Std_raw.npy'))
        ).to(self.device)

        # 2. 词向量器 + 缓存
        self.w_vectorizer = w_vectorizer
        self._token_cache = {}   # token -> (word_emb, pos_oh)   both torch Tensor

        # 3. 读 split 文件
        with cs.open(split_file, 'r') as f:
            id_list = [ln.strip() for ln in f]

        # 4. 扫描文件、建立索引
        self.data_dict, self.name_list, self.length_arr, self.max_abs_num = \
            self._build_data_dict(id_list)

        self.max_length = 20   # default pointer threshold
        self.pointer     = np.searchsorted(self.length_arr, self.max_length)
        print(f"[Dataset] Pointer at {self.pointer}, max_len={self.max_length}")

    # ──────────────────── public utils ────────────────────
    def reset_max_len(self, length: int):
        assert length <= self.max_motion_length
        self.pointer     = np.searchsorted(self.length_arr, length)
        self.max_length  = length

    def inv_transform(self, data: torch.Tensor):
        return data * self.std + self.mean

    # ──────────────────── internal ────────────────────
    def _build_data_dict(self, id_list):
        """
        基本沿用原逻辑，但：
        - 直接保留 numpy mmap path，不在这里转 torch，节省内存
        - 统计 length / max_abs_num
        """
        data_dict, length_list, new_name_list = {}, [], []
        max_abs_num = 0

        for name in tqdm(id_list, desc="scan"):
            abspath = pjoin(self.opt.abstraction_dir, f"{name}.npy")
            if not os.path.exists(abspath):
                continue
            try:
                # llama.cpp 事务: 仅记录路径 + length，getitem 再加载
                position_path   = pjoin(self.opt.position_dir, f"{name}.npy")
                motion_path     = pjoin(self.opt.motion_dir,   f"{name}.npy")
                abstraction_all_path = abspath

                # fast check length with mmap header
                motion = np.load(motion_path, mmap_mode='r')
                m_len  = len(motion)
                min_motion_len = 40 if self.opt.dataset_name == 't2m' else 24
                if m_len < min_motion_len or m_len >= 200:
                    continue

                # 读取文本
                text_file = pjoin(self.opt.text_dir, f"{name}.txt")
                with cs.open(text_file) as tf:
                    for line in tf:
                        caption, token_str, f_tag_str, to_tag_str = line.strip().split('#')
                        tokens = token_str.split(' ')
                        f_tag = float(f_tag_str) if f_tag_str else 0.
                        to_tag= float(to_tag_str) if to_tag_str else 0.

                        use_whole = (f_tag == 0. and to_tag == 0.)
                        if not use_whole:
                            # clip
                            start, end = int(f_tag*20), int(to_tag*20)
                            seg_len = end - start
                            if seg_len < min_motion_len or seg_len >= 200:
                                continue

                        # 为每条 caption 建一条记录
                        seg_name = name if use_whole else f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVW')}_{name}"
                        while seg_name in data_dict:   # 防碰撞
                            seg_name = f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVW')}_{name}"

                        data_dict[seg_name] = {
                            "motion_path"      : motion_path,
                            "position_path"    : position_path,
                            "abstraction_path" : abstraction_all_path,
                            "length"           : m_len if use_whole else seg_len,
                            "text"             : [{"caption": caption, "tokens": tokens}],
                            "f_tag20"          : None if use_whole else int(f_tag*20),
                            "to_tag20"         : None if use_whole else int(to_tag*20),
                        }
                        new_name_list.append(seg_name)
                        length_list.append(data_dict[seg_name]["length"])
                        max_abs_num = 23    # 人工固定，与原实现保持一致
            except Exception as e:
                print("[build_data_dict] error:", e, name)

        # length ascending
        name_list, length_arr = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        return data_dict, name_list, np.array(length_arr), max_abs_num

    # ──────────────────── word embedding helpers ────────────────────
    def _embed_tokens(self, tokens):
        """批量从缓存 or w_vectorizer 拉词向量，返回 (word_emb, pos_oh) 两个 [L, D] Tensor"""
        missing = [t for t in tokens if t not in self._token_cache]
        if missing:
            for tk in missing:
                we, pos = self.w_vectorizer[tk]        # numpy
                self._token_cache[tk] = (
                    torch.from_numpy(we).to(self.device),
                    torch.from_numpy(pos).to(self.device),
                )
        embs, poss = zip(*(self._token_cache[tk] for tk in tokens))
        return torch.stack(embs), torch.stack(poss)

    # ──────────────────── mask helpers ────────────────────
    def _random_mask(self, joints: torch.Tensor, n_joints: int = 22) -> torch.Tensor:
        """推理阶段：mask 指定 joint at density% 帧"""
        controllable_joints = torch.as_tensor(
            [0, 10, 11, 15, 20, 21] if n_joints == 22 else [0, 4, 7, 10, 15, 20],
            device=self.device,
        )
        choose_joint = torch.as_tensor([self.control_joint], device=self.device)
        length = joints.shape[0]

        density = self.density
        if density in [1, 2, 5]:
            choose_seq_num = density
        else:
            choose_seq_num = int(length * density / 100)
        choose_seq = torch.randperm(length, device=self.device)[:choose_seq_num]
        mask = torch.zeros((length, n_joints, 3), device=self.device, dtype=torch.bool)
        mask[choose_seq[:, None], choose_joint, :] = True

        joints_n = (joints - self.raw_mean.view(n_joints, 3)) / self.raw_std.view(n_joints, 3)
        return (joints_n * mask).view(length, -1)

    # ──────────────────── Dataset API ────────────────────
    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, idx):
        rec = self.data_dict[self.name_list[self.pointer + idx]]

        # ── 1. 载入 & to(device) ───────────────────────────
        motion_np = np.load(rec["motion_path"], mmap_mode="r")
        position_np = np.load(rec["position_path"], mmap_mode="r")
        abstraction_np = np.load(rec["abstraction_path"], allow_pickle=True)

        #   f_tag / to_tag 子段裁剪
        if rec["f_tag20"] is not None:
            motion_np      = motion_np[rec["f_tag20"] : rec["to_tag20"]]
            abstraction_np = abstraction_np[rec["f_tag20"] : rec["to_tag20"]]

        motion = torch.as_tensor(motion_np, device=self.device, dtype=torch.float32, non_blocking=True)
        position = torch.as_tensor(position_np, device=self.device, dtype=torch.float32, non_blocking=True)
        abstraction_all = torch.as_tensor(abstraction_np, device=self.device, dtype=torch.float32, non_blocking=True)

        # ── 2. 随机 caption ───────────────────────────────
        cap_dict    = random.choice(rec["text"])
        caption     = cap_dict["caption"]
        tokens      = cap_dict["tokens"]

        # maybe blank caption
        if not self.valtest and random.random() < 0.5:
            caption, tokens = "", []

        # pad / crop tokens
        if len(tokens) < self.opt.max_text_len:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"] \
                   + ["unk/OTHER"] * (self.opt.max_text_len + 2 - len(tokens) - 2)
        else:
            tokens = ["sos/OTHER"] + tokens[:self.opt.max_text_len] + ["eos/OTHER"]
        sent_len = len(tokens)

        word_embeddings, pos_one_hots = self._embed_tokens(tokens)

        # ── 3. 随机长度单位裁剪 ─────────────────────────────
        m_len = rec["length"]
        unit  = self.opt.unit_length
        m_len = ((m_len // unit) - (1 if (unit < 10 and random.random() < 1/3) else 0)) * unit
        idx0  = torch.randint(0, motion.shape[0] - m_len + 1, (1,), device=self.device).item()

        motion = motion[idx0 : idx0 + m_len]
        abstraction_seg = abstraction_all[idx0 : idx0 + m_len]

        # ── 4. abstraction 随机扰动(全 GPU) ─────────────────
        abstraction_seg = sample_abstraction_all(abstraction_seg)               # (t, k, 8, 3)
        abstraction_seg = apply_random_global_scaling_translation(
            abstraction_seg, scaling_strength=0.3, translation_strength=0.0
        )
        if random.random() < 0.5:
            abstraction_seg = apply_random_transformation_full(
                abstraction_seg, rotation_strength=np.pi * 0.1, translation_strength=0.05
            )
        if random.random() < 0.5:
            abstraction_seg = apply_random_drop_full(abstraction_seg)

        # shuffle k-dimension
        k = abstraction_seg.shape[1]
        abstraction_seg = abstraction_seg[:, torch.randperm(k, device=self.device)]
        abstraction_seg = abstraction_seg.view(abstraction_seg.shape[0], -1)    # (t, k*8*3)

        # ── 5. motion 归一化 & 补零到 max_motion_length ───
        motion = (motion - self.mean) / self.std

        pad_len = self.max_motion_length - m_len
        if pad_len > 0:
            motion = torch.cat([motion,
                                torch.zeros(pad_len, motion.shape[1], device=self.device)])
            abstraction_seg = torch.cat([abstraction_seg,
                                torch.zeros(pad_len, abstraction_seg.shape[1], device=self.device)])

        # ── 6. joints mask (可选) ─────────────────────────
        # n_joints = 22 if motion.shape[-1] == 263 else 21
        # hint = self._random_mask(joints, n_joints)

        tokens_joined = "_".join(tokens)

        return (word_embeddings, pos_one_hots, caption, sent_len,
                motion, m_len, tokens_joined, abstraction_seg, self.max_abs_num)

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
    def __init__(self, mode, datapath='./dataset/humanml_opt.txt', split="train", control_joint=0, density=100, **kwargs):
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
        opt.meta_dir = './dataset'
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
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean, self.std, self.split_file, self.w_vectorizer, mode, control_joint, density)
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