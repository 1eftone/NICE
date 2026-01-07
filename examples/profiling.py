import sys
sys.path.append('../')

import argparse
from scoop.utils import construct_labels
from scoop.load_ascad_data import FastTensorDataLoader
import numpy as np
import torch
import h5py
import math
import matplotlib
matplotlib.use('Agg')  # 必须在 import pyplot 之前设置
import matplotlib.pyplot as plt
from scoop.scoop import Scoop
#from scoop.custom_mlp import MLPModel
from scoop.train_model import train_model
#from scoop.transformer import EstraNet
from scoop.estranet import EstraNet  # 假设你把上面的代码存为了 scoop/estranet.py
import os

# --- [新增] 参数解析部分 ---
parser = argparse.ArgumentParser(description='ASCADv2 Profiling with EstraNet')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--pe_type', type=str, default='estranet', choices=['estranet', 'absolute', 'none'], help='Positional Encoding Type')
parser.add_argument('--save_dir', type=str, default='/scratch/e240023/scoop_experiments/', help='Directory to save results')
parser.add_argument('--dataset_path', type=str, default='/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5', help='Path to dataset')
parser.add_argument('--n_epochs', type=int, default=25, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
from subprocess import call
# call(["nvcc", "--version"]) does not work
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Active CUDA Device:", device)


path = '/home/joey1/Documents/joey/Data/ASCAD/ascadv2-extracted.h5'

# --- 2. 准备路径与目录 ---
os.makedirs(args.save_dir, exist_ok=True)

# 动态生成唯一的文件名，防止实验互相覆盖
run_tag = f"{args.pe_type}_seed{args.seed}"
path_final = os.path.join(args.save_dir, f'ascadv2_final_{run_tag}.pt')
path_best_loss = os.path.join(args.save_dir, f'ascadv2_best_loss_{run_tag}.pt')
path_best_rank = os.path.join(args.save_dir, f'ascadv2_best_rank_{run_tag}.pt')
log_file_csv = os.path.join(args.save_dir, f'training_log_{run_tag}.csv')
model_save_log_txt = os.path.join(args.save_dir, f'model_checkpoints_{run_tag}.log')

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
plt.plot(profiling_traces['traces'][199999])
# 将原代码中的 plt.show() 替换为:
plt.savefig('trace_plot-199999.png')  # 保存图片代替显示

idx_to_removes = [99999, 199999, 299999, 399999, 499999] # due to bug

metadata_profiling = profiling_traces['metadata']
inds = metadata_profiling['key']

byte = 4
labels, pts, keys = construct_labels(byte, profiling_traces)
labels_attack, pts_attack, keys_attack = construct_labels(byte, attack_traces)


train_len = 200000
test_len = 10000

X = torch.Tensor(np.array(profiling_traces['traces'][:train_len]))
Y = torch.LongTensor(labels[:train_len])

# removing the 0-traces of the ascadv2_extracted dataset
X[99999] = X[99998]
X[199999] = X[199998]
Y[99999] = Y[99998]
Y[199999] = Y[199998]

X_attack = torch.Tensor(np.array(attack_traces['traces'][:test_len]))
Y_attack = torch.LongTensor(labels_attack[:test_len])

n_classes = 256
signal_length = len(X[0])
lr = 1e-4
beta1 = 0.965
beta2 = 0.999
n_epochs = 25
activation = 'ReLU'
n_linear = 1
linear_size = math.ceil(signal_length+n_classes*(2/3))
input_bn = True
dense_bn = False
batch_size = 128
weight_decay = 0.2

train_loader = FastTensorDataLoader(X, Y, batch_size=batch_size, shuffle=True)

attack_loader = FastTensorDataLoader(X_attack, Y_attack, batch_size=batch_size, shuffle=False)


# 这里需要提取对应的 metadata
metadata_attack = attack_traces['metadata']
pts_attack_source = metadata_attack['plaintext']

# 提取第3个字节 (byte=2) 或者第4个字节 (byte=3)，取决于 construct_labels 里的 byte 变量
# 假设前面你定义了 byte = 2 (或 4)
# 注意：PyTorch 需要 Tensor 类型
val_plaintexts = torch.tensor([pts_attack_source[i][byte] for i in range(len(pts_attack_source))]).long()

# 2. 替换模型初始化代码
# --- 原代码 ---
#model = MLPModel(signal_length=signal_length, n_classes=n_classes, n_linear=n_linear, linear_size=linear_size, activation=activation, input_bn=input_bn, dense_bn=dense_bn).to(device)

# --- 修改后 ---
# 注意：EstraNet 的超参数可能需要调整，Transformer 通常需要更小的 lr 或者 warmup
# model = EstraNet(
#     d_model=128,      # 可调
#     n_head=8,         # 可调
#     n_layers=2,       # 可调
#     num_classes=256   # 对应 ASCAD 的 byte 值域
# ).to(device)
# 实例化 EstraNet
# 注意：signal_length 是原始迹线长度，EstraNet 内部会自动处理 Conv 后的长度变化
model = EstraNet(n_classes=256, signal_length=signal_length, d_model=128, pe_type=args.pe_type).to(device)

#optimizer = Scoop(model.parameters(), lr=args.lr, betas=(beta1, beta2), weight_decay=weight_decay, hessian_iter=1, estimator='low_variance')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 3. 调用训练函数
train_loss, val_loss, _ = train_model(
    model, 
    optimizer, 
    n_epochs, 
    train_loader, 
    attack_loader, # 这里传入的就是包含 1w 条数据的验证集 loader
    val_plaintexts=val_plaintexts, # <--- 关键：传入明文
    verbose=True, 
    path=path_final,            # 退出时保存
    best_path=path_best_loss,   # Loss 最低时保存
    best_rank_path=path_best_rank, # Rank 最低时保存 (新增)
    device=device, 
    MLP=False, 
    finetuning=False, 
    entropy=8,
    log_file=f'training_log_seed{seed}.csv',
    model_save_log=model_save_log_txt # <--- 独立的 Log 文件
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# train_loss = np.load('ascadv2_singlemlp_train_loss.npy')
# val_loss = np.load('ascadv2_singlemlp_val_loss.npy')

fig, ax = plt.subplots()
ax.plot(train_loss, label='Train loss', color='k', linestyle='--')
ax.plot(val_loss, label='Validation loss', color='k')

ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
formatter = FuncFormatter(lambda y, _: f"{y:.4f}")
ax.yaxis.set_major_formatter(formatter)

ax.legend()
plt.title(f'ASCADv2 {args.pe_type} loss')
# plt.ylim(7.9995, 8.0001)
plt.grid(linestyle='--')
plt.savefig(f'sophia{seed}.png')
# plt.show()

