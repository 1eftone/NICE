import torch
import numpy as np
from tqdm import tqdm # 导入进度条库

def guessing_entropy(model, X_attack, plaintexts, sbox, device, correct_key, n_bits=8, n_tries=10, n_attack=300000, verbose=False, step=10000, batch_pred=True):
    """
    [GPU 加速版 + 进度条] Guessing Entropy 计算
    保持接口不变，内部逻辑优化，显存安全。
    """
    model.eval()
    
    # 1. 数据预处理
    n_traces = min(len(X_attack), n_attack)
    X_attack = X_attack[:n_traces]
    
    if not isinstance(plaintexts, torch.Tensor):
        plaintexts = torch.tensor(plaintexts[:n_traces], dtype=torch.long, device=device)
    else:
        plaintexts = plaintexts[:n_traces].to(device)
        
    if not isinstance(sbox, torch.Tensor):
        sbox = torch.tensor(sbox, device=device)
    else:
        sbox = sbox.to(device)

    # --- 2. 批量推理 (Inference) 带进度条 ---
    # 这里是原本最耗时的部分
    if verbose: 
        print(f"Pre-computing predictions for {n_traces} traces...")
    
    # 【显存安全配置】Batch Size = 128
    batch_size = 64 
    all_preds = []
    
    # 使用 range 生成迭代器
    iterator = range(0, n_traces, batch_size)
    
    # 如果 verbose=True，则包裹 tqdm 显示进度条
    if verbose:
        iterator = tqdm(iterator, desc="Inference", unit="batch")

    with torch.no_grad():
        for i in iterator:
            batch_x = X_attack[i : i + batch_size].to(device)
            if batch_x.dim() == 2: batch_x = batch_x.unsqueeze(1)
            
            # 模型输出 Log 概率
            all_preds.append(model(batch_x))
            
    Y_probs = torch.cat(all_preds, dim=0) # [N, 256]
    
    # --- 3. 密钥概率矩阵构建 ---
    if verbose: print("Computing Key Probabilities...")

    keys = torch.arange(256, device=device).unsqueeze(0) # [1, 256]
    pts = plaintexts.unsqueeze(1) # [N, 1]
    
    # SBox 查表
    hyp_labels = sbox[pts ^ keys].long() # [N, 256]
    
    # 提取概率
    key_log_probs = torch.gather(Y_probs, 1, hyp_labels) # [N, 256]
    
    # 释放显存
    del Y_probs, hyp_labels, all_preds
    torch.cuda.empty_cache()

    # --- 4. 累积求和与排序 (Cumsum) 带进度条 ---
    # 这里对应原本的 n_tries 循环
    
    accumulated_ge = torch.zeros(n_traces, device=device)
    
    # 设置进度条
    try_iterator = range(n_tries)
    if verbose:
        try_iterator = tqdm(try_iterator, desc="Calculating GE", unit="try")
    
    for _ in try_iterator:
        # 打乱
        perm = torch.randperm(n_traces, device=device)
        shuffled = key_log_probs[perm]
        
        # 累积求和 (核心加速)
        cum_probs = torch.cumsum(shuffled, dim=0)
        
        # 计算 Rank
        score_correct = cum_probs[:, correct_key].unsqueeze(1)
        ranks = (cum_probs > score_correct).sum(dim=1).float() + 1
        accumulated_ge += ranks

    # --- 5. 结果处理 ---
    final_ge = (accumulated_ge / n_tries).cpu().numpy()
    n_attacks = np.arange(1, n_traces + 1)
    
    # 采样返回 (为了兼容 step 参数)
    if step > 1:
        indices = np.arange(step-1, n_traces, step)
        # 边界检查，防止 indices 为空
        if len(indices) == 0:
             return final_ge, n_attacks
        return final_ge[indices], n_attacks[indices]
        
    return final_ge, n_attacks