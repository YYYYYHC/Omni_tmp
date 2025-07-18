# This code is modified based on https://github.com/GuyTevet/motion-diffusion-model
import numpy as np
import torch
import torch.nn as nn
import clip
from model.rotation2xyz import Rotation2xyz
from .transformer import *


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class CMDM(torch.nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, max_abs_num=None, *args, **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset

        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)

        self.input_feats = self.njoints * self.nfeats

        self.normalize_output = kargs.get('normalize_encoder_output', False)

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.emb_trans_dec = emb_trans_dec

        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
        # --- MDM ---
        self.input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                num_layers=self.num_layers)

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                print('EMBED TEXT')
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints,
                                            self.nfeats)
        # ------
        # --- CMDM ---
        # input 263 or 6 * 3 or 3
        n_sample = 5 if njoints == 263 else 21
        self.input_hint_block = HintBlock(self.data_rep, max_abs_num * 3 *8, self.latent_dim)

        self.c_input_process = InputProcess(self.data_rep, self.input_feats+self.gru_emb_dim, self.latent_dim)

        self.c_sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        print("TRANS_ENC init")
        seqTransEncoderLayer = TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.c_seqTransEncoder = TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=self.num_layers,
                                                    return_intermediate=True)

        self.zero_convs = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]))
        
        self.c_embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.c_embed_text = nn.Linear(self.clip_dim, self.latent_dim)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
        return self.clip_model.encode_text(texts).float()

    def cmdm_forward(self, x, timesteps, y=None, weight=1.0):
        """
        Realism Guidance
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        emb = self.c_embed_timestep(timesteps)  # [1, bs, d]

        seq_mask = y['hint'].sum(-1) != 0

        guided_hint = self.input_hint_block(y['hint'].float())  # [bs, d]
        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.c_embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        x = self.c_input_process(x)
    
        x += guided_hint * seq_mask.permute(1, 0).unsqueeze(-1)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.c_sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.c_seqTransEncoder(xseq)  # [seqlen+1, bs, d]

        control = []
        for i, module in enumerate(self.zero_convs):
            control.append(module(output[i]))
        control = torch.stack(control)

        control = control * weight
        return control
    
    def mdm_forward(self, x, timesteps, y=None, control=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        emb = self.embed_timestep(timesteps)  # [1, bs, d]

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        x = self.input_process(x)

        # adding the timestep embed
        xseq = torch.cat((emb, x), axis=0)  # [seqlen+1, bs, d]
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen+1, bs, d]
        output = self.seqTransEncoder(xseq, control=control)[1:]  # , src_key_padding_mask=~maskseq)  # [seqlen, bs, d]

        output = self.output_process(output)  # [bs, njoints, nfeats, nframes]
        return output

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        if 'hint' in y.keys():
            control = self.cmdm_forward(x, timesteps, y)
        else:
            n_samples = 5 if self.njoints == 263 else None
            y_ = {'hint': torch.zeros((x.shape[0], x.shape[-1], n_samples * 2 * 3), device=x.device)}
            y_.update(y)
            control = self.cmdm_forward(x, timesteps, y_)
        output = self.mdm_forward(x, timesteps, y, control)
        return output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)


    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)

   

class HintBlock(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.dummy = nn.Parameter(torch.zeros(1, 1, latent_dim//2))   # placeholder for empty abstraction
        self.dummy.requires_grad = False
        self.latent_dim = latent_dim
        self.point_dim  = latent_dim // 2
        self.group_size = 24  # 每组24个元素
        self.num_groups = input_feats // self.group_size
        assert input_feats % self.group_size == 0, "input_feats must be divisible by group_size"
        
        # 每组特征的MLP
        # self.input_proj = nn.Sequential(
        #     nn.Linear(self.group_size, self.latent_dim // 2),
        # )
        # 1) φ(·) —— 对单个顶点 (x,y,z) 的 MLP
        self.point_mlp = nn.Sequential(
            nn.Linear(3, self.point_dim//2),           # 3 → d
            nn.GELU()
        )
        # 2) ψ(·) —— 聚合之后再映射（保留你的名字 input_proj）
        self.input_proj = nn.Sequential(
            nn.Linear(self.point_dim//2, self.point_dim),
            nn.LayerNorm(self.point_dim),
            nn.GELU()
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim // 2,
            nhead=4,
            dim_feedforward=self.latent_dim,
            dropout=0.2,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.final_proj = nn.Sequential(
            nn.Linear(self.latent_dim // 2 , self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
            zero_module(nn.Linear(self.latent_dim, self.latent_dim))
        )

    def forward(self, x):
        # x: [b, t, f_in]
        b, t, f = x.shape
        
        # 将特征按组划分
        x = x.view(b, t, self.num_groups, self.group_size)  # [b, t, num_groups, group_size]
        # 创建padding mask
        padding_mask = (x.sum(dim=-1) == 0)  # [b, t, num_groups]
        # project group features
        pts = x.view(b * t * self.num_groups, 8, 3)
        # ② φ：逐点独立映射  -> [B*T*N, 8, d]
        point_feat = self.point_mlp(pts)
        # ③ 聚合 (sum / mean / max 都可；这里用 mean) -> [B*T*N, d]
        global_feat = point_feat.mean(dim=1) + point_feat.max(dim=1).values
        
        # ④ ψ：再做一次线性映射 -> [B*T*N, d]
        global_feat = self.input_proj(global_feat)

        # ⑤ reshape 回原维度
        global_feat = global_feat.view(b, t, self.num_groups, -1)
        # group_features = self.input_proj(x)  # [b, t, num_groups, latent_dim//2]
        
        # 重组特征用于transformer处理
        group_features = global_feat.reshape(b * t, self.num_groups, -1)  # [b*t, num_groups, latent_dim//2]
        
        
        padding_mask = padding_mask.reshape(b * t, self.num_groups)  # [b*t, num_groups]
        # 将padding位置的特征设为0
        # group_features = group_features.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Transformer处理
        transformed = self.transformer(group_features)  # [b*t, num_groups, latent_dim//2]
        
        # 重组回原始维度
        transformed[padding_mask] = self.dummy
        
        valid_mask = (~padding_mask).unsqueeze(-1)        # [B*T, N, 1]
        seq_valid_mask = ~(valid_mask.sum(1) == 0).squeeze(-1)
        sum_feat   = (transformed * valid_mask).sum(1)
        cnt_feat   = valid_mask.sum(1)
        mean_feat  = sum_feat / (cnt_feat+1)              # [B*T, d]
        mean_feat[~seq_valid_mask] = 0
        
        max_feat  = transformed.amax(dim=1)        # 如果想保多信息
        seq_feat = mean_feat + max_feat                 # or torch.cat([...], dim=-1)

        # 展平并投影到目标维度
        seq_feat = seq_feat.reshape(b, t, -1)  # [b, t, latent_dim//2]
        output = self.final_proj(seq_feat)  # [b, t, latent_dim]
        
        output = output.permute(1, 0, 2)  # [t, b, latent_dim]
        
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output