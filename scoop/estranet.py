import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 组件 1: 线性注意力机制 (O(N) Complexity) ---
# (保持不变)
class LinearSelfAttention(nn.Module):
    def __init__(self, d_model, d_head, n_head, d_kernel_map=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.d_kernel_map = d_kernel_map
        
        self.value_weight = nn.Linear(d_model, n_head * d_head, bias=False)
        self.output_weight = nn.Linear(n_head * d_head, d_model, bias=False)
        self.output_dropout = nn.Dropout(dropout)
        
        self.pos_ft_weight = nn.Parameter(torch.randn(d_model, n_head * d_head), requires_grad=False)
        self.pos_ft_scale = nn.Parameter(torch.ones(1, 1, n_head, 1))
        
        self.pos_ft_offsets = nn.Parameter(torch.zeros(1, 1, n_head, 1))
        self._init_offsets()

        self.register_buffer('projection_matrix', self._gen_projection_matrix(d_kernel_map, d_head))

    def _init_offsets(self):
        head_range = 1.0 / self.n_head
        offsets = torch.arange(0, 1.0, head_range) + head_range / 2.0
        self.pos_ft_offsets.data = offsets.view(1, 1, self.n_head, 1)

    def _gen_projection_matrix(self, m, d):
        blocks = []
        for _ in range(m // d):
            block = torch.randn(d, d)
            q, _ = torch.linalg.qr(block)
            blocks.append(q.T)
        rem = m % d
        if rem > 0:
            block = torch.randn(d, d)
            q, _ = torch.linalg.qr(block)
            blocks.append(q.T[:rem])
        mat = torch.vstack(blocks)
        multiplier = torch.norm(torch.randn(m, d), dim=1, keepdim=True)
        return mat * multiplier

    def _fourier_kernel(self, data):
        data_normalizer = 1.0 / (data.shape[-1] ** 0.25)
        ratio = 1.0 / (self.projection_matrix.shape[0] ** 0.5)
        data = data_normalizer * data
        data_dash = torch.einsum("blhd,md->blhm", data, self.projection_matrix)
        data_sin = ratio * torch.sin(data_dash)
        data_cos = ratio * torch.cos(data_dash)
        return torch.cat([data_sin, data_cos], dim=-1)

    def forward(self, x, pos_ft, pos_ft_slopes):
        B, L, _ = x.shape
        v = self.value_weight(x).view(B, L, self.n_head, self.d_head)
        
        pos_ft_proj = torch.einsum("blm,md->bld", pos_ft, self.pos_ft_weight).view(B, L, self.n_head, self.d_head)
        slope_proj = torch.einsum("blm,md->bld", pos_ft_slopes, self.pos_ft_weight).view(B, L, self.n_head, self.d_head)
        
        query_pos = self.pos_ft_scale * pos_ft_proj
        key_pos = query_pos + self.pos_ft_offsets * (self.pos_ft_scale * slope_proj)
        
        q_prime = self._fourier_kernel(query_pos)
        k_prime = self._fourier_kernel(key_pos)
        
        kv = torch.einsum("blhm,blhd->bhmd", k_prime, v)
        z = torch.einsum("blhm,bhmd->blhd", q_prime, kv)
        
        norm_factor = torch.norm(slope_proj, dim=-1, keepdim=True) / float(L)
        z = z * norm_factor
        
        z = z.reshape(B, L, -1)
        return self.output_dropout(self.output_weight(z))

# --- 组件 2: 位置特征生成器 ---
# (保持不变)
class PositionalFeature(nn.Module):
    def __init__(self, d_model, beta_hat_2=100.0):
        super().__init__()
        self.d_model = d_model
        slopes = torch.arange(d_model, 0, -4.0) / d_model * beta_hat_2
        self.register_buffer('slopes', slopes)

    def forward(self, L, device):
        pos_seq = torch.arange(0, L, 1.0, device=device)
        normalized_slopes = self.slopes / float(L - 1)
        forward = torch.outer(pos_seq, normalized_slopes)
        backward = torch.flip(forward, dims=[0])
        pos_ft = torch.cat([forward, backward, -forward, -backward], dim=-1)
        slope_rep = normalized_slopes.unsqueeze(0).repeat(L, 1)
        pos_ft_slopes = torch.cat([slope_rep, -slope_rep, -slope_rep, slope_rep], dim=-1) * float(L - 1)
        return pos_ft, pos_ft_slopes

# --- 组件 3: Feed Forward Network (新增) ---
# 【关键新增】补全 Transformer 的 FFN 部分
# --- 组件 3: Feed Forward Network (新增) ---
# 【关键新增】补全 Transformer 的 FFN 部分
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# --- 组件 4: 池化多头注意力 (PMA) 模块 ---
class PMA(nn.Module):
    def __init__(self, d_model, num_heads, k_seeds=1, dropout=0.1):
        """
        参数:
            d_model: 输入特征维度 (hidden size)
            num_heads: 注意力头数
            k_seeds: 种子向量的数量 (类似于输出的特征点数量)。
                     如果设为 1，效果类似于 [CLS] token；
                     如果设为 >1 (如 4)，可以捕获多种不同的泄漏模式。
            dropout: Dropout 比率
        """
        super().__init__()
        self.d_model = d_model
        self.k_seeds = k_seeds
        
        # 1. 定义可学习的种子向量 S (Query)
        # Shape: (1, k_seeds, d_model)
        self.seeds = nn.Parameter(torch.randn(1, k_seeds, d_model))
        
        # 2. 多头注意力机制 (MHA)
        # batch_first=True 使得输入输出格式为 (Batch, Seq, Feature)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # 3. 前馈网络 (Feed Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        
        # 4. Layer Normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        输入 x: (Batch_Size, Sequence_Length, d_model)
        """
        batch_size = x.size(0)
        
        # 1. 广播种子向量以匹配 Batch 大小
        # seeds: (Batch_Size, k_seeds, d_model)
        seeds = self.seeds.repeat(batch_size, 1, 1)
        
        # 2. Cross-Attention
        # Query = Seeds, Key = x, Value = x
        # attn_out: (Batch_Size, k_seeds, d_model)
        attn_out, _ = self.mha(query=seeds, key=x, value=x)
        
        # 3. 残差连接 + LayerNorm
        out = self.ln1(seeds + attn_out)
        
        # 4. Feed Forward + 残差连接 + LayerNorm
        ff_out = self.ffn(out)
        out = self.ln2(out + ff_out)
        
        # 5. 输出处理
        # 现在的 out 是 (Batch, k_seeds, d_model)
        # 我们需要将其展平以便送入分类器: (Batch, k_seeds * d_model)
        return out.reshape(batch_size, -1)
    

    
# --- 主模型: EstraNet Wrapper (Fixed) ---
class EstraNet(nn.Module):
    def __init__(self, n_classes, signal_length, d_model=128, n_head=4, n_layers=2, 
                 pe_type='estranet'): 
        super().__init__()
        self.pe_type = pe_type
        self.d_model = d_model
        
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_head ({n_head})")
        self.d_head = d_model // n_head

        # 1. Stem Layer
        # self.stem = nn.Sequential(
        #     nn.Conv1d(1, 32, kernel_size=11, padding=5),
        #     nn.ReLU(),
        #     nn.AvgPool1d(2),
        #     nn.Conv1d(32, d_model, kernel_size=11, padding=5),
        #     nn.ReLU(),
        #     nn.AvgPool1d(2)
        # )
        # self.stem_len = signal_length // 4 

        # Stem Layer
        self.stem = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=300, stride=100, padding=100),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )
        self.stem_len = signal_length // 100 

        # 2. 位置编码模块
        if pe_type == 'estranet':
            self.pos_feature_gen = PositionalFeature(d_model)
            # 【修正点 1】: PositionalFeature 内部使用了 stride=4 采样再拼接，
            # 最终输出维度其实等于 d_model，而不是 4*d_model
            self.pos_dim = d_model  
        elif pe_type == 'absolute':
            self.abs_pos_emb = nn.Parameter(torch.randn(1, self.stem_len, d_model) * 0.02)
        
        # 3. Encoder Layers
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            if pe_type == 'estranet':
                # 手动组装完整的 Transformer Block
                layer_block = nn.ModuleDict({
                    'attn': LinearSelfAttention(d_model, d_head=self.d_head, n_head=n_head),
                    'norm1': nn.LayerNorm(d_model),
                    'ffn': FeedForward(d_model, dim_feedforward=d_model*4),
                    'norm2': nn.LayerNorm(d_model)
                })
                # 【修正点 2】: 使用修正后的 self.pos_dim (即 128) 初始化权重
                # 注意：其实 LinearSelfAttention 默认就是 d_model，这里显式覆盖是为了保险
                layer_block['attn'].pos_ft_weight = nn.Parameter(
                    torch.randn(self.pos_dim, n_head * self.d_head), requires_grad=False)
                self.layers.append(layer_block)
            else:
                layer = nn.TransformerEncoderLayer(d_model, nhead=n_head, batch_first=True,
                activation='gelu',
                norm_first=True)
                self.layers.append(layer)

        self.use_pma = True # 开关方便做消融实验
        k_seeds = 4    # PMA 的种子向量数量
        self.k_seeds = k_seeds
        
        if self.use_pma:
            self.pooler = PMA(d_model=d_model, num_heads=4, k_seeds=k_seeds)
            
            # 注意：如果 k_seeds > 1，输出维度变了！
            # PMA 输出是 (Batch, k_seeds, d_model) -> flatten -> (Batch, k_seeds * d_model)
            classifier_input_dim = d_model * k_seeds
        else:
            classifier_input_dim = d_model

        # --- [修改 2] Classifier 的输入维度需要动态调整 ---
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, d_model), # 这里的输入维度变了
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # 维度处理
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        
        x = self.stem(x)   # [B, d_model, L']
        x = x.permute(0, 2, 1) # [B, L', d_model]
        
        B, L, D = x.shape
        
        if self.pe_type == 'estranet':
            pos_ft, pos_slopes = self.pos_feature_gen(L, x.device)
            pos_ft = pos_ft.unsqueeze(0).expand(B, -1, -1)
            pos_slopes = pos_slopes.unsqueeze(0).expand(B, -1, -1)
            
            for layer in self.layers:
                # 1. Attention Block
                residual = x
                out = layer['attn'](x, pos_ft, pos_slopes)
                x = layer['norm1'](x + out) 
                
                # 2. FFN Block
                residual = x
                out = layer['ffn'](x)
                x = layer['norm2'](x + out)
        
        elif self.pe_type == 'absolute':
            actual_len = min(L, self.abs_pos_emb.shape[1])
            pe = self.abs_pos_emb[:, :actual_len, :]
            if x.size(1) != pe.size(1):
                # 取两者中较小的长度
                min_len = min(x.size(1), pe.size(1))
            
                # 裁剪 x 和 pe 到相同长度
                x = x[:, :min_len, :]
                pe = pe[:, :min_len, :]
            x = x + pe
            for layer in self.layers:
                x = layer(x)
        
        elif self.pe_type == 'none':
            for layer in self.layers:
                x = layer(x)

        # PMA selection
        if self.use_pma:
            # --- 使用 PMA 聚合 ---
            # Output: [Batch, k_seeds * d_model]
            x = self.pooler(x) 
        else:
            # --- 原本的 Average Pooling ---
            x = x.mean(dim=1) 

        # 送入分类器
        return self.classifier(x)
    

