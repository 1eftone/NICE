import sys
import os

# 1. 设置运行路径和绘图后端
sys.path.append('../')  # 确保能调用 scoop 包
import matplotlib
matplotlib.use('Agg')   # 服务器 Headless 模式专用，必须在 import pyplot 前
import matplotlib.pyplot as plt

from scoop.utils import construct_labels, sbox
from scoop.metrics import guessing_entropy
#from scoop.custom_mlp import MLPModel
#from scoop.transformer import EstraNet
from scoop.estranet import EstraNet  # 假设你把上面的代码存为了 scoop/estranet.py
import h5py
import torch
import numpy as np
import math
import argparse


# --- [新增] 参数解析 ---
parser = argparse.ArgumentParser(description='ASCADv2 Attack Evaluation')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--pe_type', type=str, required=True, choices=['estranet', 'absolute', 'none'], help='Must match the training configuration')
parser.add_argument('--model_path', type=str, required=True, help='Full path to the .pt model file to evaluate')
parser.add_argument('--dataset_path', type=str, default='/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5')
args = parser.parse_args()

# --- 打印环境信息 ---
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Active CUDA Device:", device)

# -------------------------------------------------------------------------
# 配置路径 (请根据实际情况修改)
# -------------------------------------------------------------------------

path = '/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5'

# --- 3. 加载数据 (使用 args.dataset_path) ---
print(f"Loading data from {args.dataset_path}...")
try:
    ascad_db = h5py.File(args.dataset_path, 'r')
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

profiling_traces = ascad_db['Profiling_traces']
attack_traces = ascad_db['Attack_traces']

X_test = attack_traces['traces']
metadata_test = attack_traces['metadata']
pt_test = metadata_test['plaintext']
key_test = metadata_test['key']
Y_test = attack_traces['labels']

plt.figure()
plt.plot(profiling_traces['traces'][0])
# 将原代码中的 plt.show() 替换为:
plt.savefig('trace0')  # 保存图片代替显示

metadata_profiling = profiling_traces['metadata']
inds = metadata_profiling['key']

byte = 4

labels, pts, keys = construct_labels(byte, profiling_traces)
labels_attack, pts_attack, keys_attack = construct_labels(byte, attack_traces)

train_len = 200000
test_len = 300000
val_len = 1000
X = torch.Tensor(np.array(profiling_traces['traces'][:train_len]))
Y = torch.LongTensor(labels[:train_len])

X_val = torch.Tensor(np.array(attack_traces['traces']))
Y_val = torch.LongTensor(labels_attack)
X_attack = torch.Tensor(np.array(profiling_traces['traces'][train_len:train_len+test_len]))
Y_attack = torch.LongTensor(labels[train_len:train_len+test_len])


n_classes = 256
signal_length = len(X[0])
lr = 1e-3
beta1 = 0.965
beta2 = 0.999
n_epochs = 25
activation = 'ReLU'
n_linear = 1
linear_size = math.ceil(signal_length+n_classes*(2/3))
input_bn = True
dense_bn = False
batch_size = 32
weight_decay = 0.2

# 2. 替换模型初始化代码
# --- 原代码 ---
#best_model = MLPModel(signal_length=signal_length, n_classes=n_classes, n_linear=n_linear, linear_size=linear_size, activation=activation, input_bn=input_bn, dense_bn=dense_bn).to(device)

# --- 修改后 ---
# 注意：EstraNet 的超参数可能需要调整，Transformer 通常需要更小的 lr 或者 warmup
# best_model = EstraNet(
#     d_model=128,      # 可调
#     n_head=8,         # 可调
#     n_layers=2,       # 可调
#     num_classes=256   # 对应 ASCAD 的 byte 值域
# ).to(device)
# 实例化 EstraNet
# 注意：signal_length 是原始迹线长度，EstraNet 内部会自动处理 Conv 后的长度变化
best_model = EstraNet(n_classes=256, signal_length=signal_length, d_model=128, pe_type=args.pe_type).to(device)

# 之后直接传给 scoop.train_model 即可，接口完全兼容



metadata_profiling = profiling_traces['metadata']
pts_attack = metadata_profiling['plaintext']
keys_attack = metadata_profiling['key']
val_len = train_len

# Load the model

best_model.load_state_dict(torch.load(args.model_path, map_location=device))
best_model.to(device)
n_params = sum(p.numel() for p in best_model.parameters())
print('Number of parameters:', n_params)    

pts = np.array([pts_attack[i][byte] for i in range(len(pts_attack))]).astype(int)
pts = pts[val_len:val_len+test_len]
print(len(pts))

keys_attack = np.array([keys_attack[i][byte] for i in range(len(keys_attack))]).astype(int)
keys_attack = keys_attack[val_len:val_len+test_len]
new_pts = np.array([pts[i]^keys_attack[i] for i in range(len(pts))]).astype(int)
true_key = int(0)

# removing 0-traces of the ascadv2_extracted dataset

X_attack[99999] = X_attack[99998]
X_attack[199999] = X_attack[199998]
X_attack[299999] = X_attack[299998]
new_pts[99999] = new_pts[99998]
new_pts[199999] = new_pts[199998]
new_pts[299999] = new_pts[299998]
plt.figure()
plt.plot(X_attack[299999])

n_tries = 20 # number of times GE is computed, higher is more statistically significant

GE_scoop, n_attacks =  guessing_entropy(best_model, X_attack, new_pts, sbox, device, true_key, n_bits=8, n_tries=n_tries, n_attack=300000, verbose=True, step=10000, batch_pred=True)

# np.save('GE_scoop.npy', GE_scoop)
# np.save('n_attacks.npy', n_attacks)

fig, ax = plt.subplots()
ax.plot(n_attacks, GE_scoop*2, color='k')
ax.set_xlabel('Number of traces')
ax.set_ylabel('Guessing entropy')
ax.grid(linestyle='--')
#plt.savefig('guessing_entropy.pdf')
# 将原代码中的 plt.show() 替换为:
# 可以根据 model_path 的名字来命名结果图片，避免覆盖
model_filename = os.path.basename(args.model_path).replace('.pt', '')
save_plot_path = f'ge_result_{model_filename}.png'
plt.savefig(save_plot_path)
print(f"Result saved to {save_plot_path}")