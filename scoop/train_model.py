import torch
import torch.nn.functional as F
import math
import time
import numpy as np
import csv
import os
import datetime

# --- SBox (用于计算假设中间值) ---
SBOX = torch.tensor([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
], dtype=torch.long)

def compute_key_rank(Y_pred_log, plaintexts, device, correct_key=0):
    """
    计算基于 Key 的累积 Rank。
    Y_pred_log: 模型输出的 Log Probabilities (Batch, 256)
    plaintexts: 当前 Batch 的明文 (Batch,)
    """
    # 确保 SBOX 在正确的设备上
    sbox = SBOX.to(device)
    
    # 1. 扩展维度以计算所有 256 个 Key 的假设
    # Batch size
    B = Y_pred_log.shape[0]
    
    # 生成所有候选 key: [0, 1, ..., 255] -> (1, 256)
    keys = torch.arange(256, device=device).unsqueeze(0) 
    
    # 扩展 plaintexts: (B, 1)
    pts = plaintexts.unsqueeze(1)
    
    # 2. 计算假设中间值 Hypothesis: SBox(pt ^ key)
    # broadcasting: (B, 1) ^ (1, 256) -> (B, 256)
    hyp_intermediates = sbox[pts ^ keys] # (B, 256)
    
    # 3. 从预测结果中提取概率
    # Y_pred_log: (B, 256_classes)
    # 我们需要对于每个 key guess，提取其对应的 intermediate label 的概率
    # gather dim=1. hyp_intermediates 就是我们要取出的 index
    batch_key_probs = Y_pred_log.gather(1, hyp_intermediates) # (B, 256)
    
    # 4. 返回 Batch 内的累加 (Sum over Batch)
    return batch_key_probs.sum(dim=0) # (256,)


def train_model(model, optimizer, n_epochs, train_loader, valid_loader, 
                val_plaintexts=None, # [NEW] 传入验证集的 Plaintext
                hessian_update=10, verbose=False, save_best_model=True, 
                path='final_model.pt', best_path='best_loss_model.pt', best_rank_path='best_rank_model.pt', 
                device=None, finetuning=False, entropy=8, MLP=False, 
                log_file='training_log.csv', model_save_log='model_checkpoints.log'):
    
    # 自动判断是否需要 Hessian (适配 Adam vs Scoop)
    use_hessian = hasattr(optimizer, 'hutchinson_hessian')
    if verbose:
        print(f"Optimizer Type: {type(optimizer).__name__}")
        print(f"Use Hessian Update: {use_hessian}")
    
    train_losses = []
    valid_losses = []
    
    best_val_loss = float('inf') 
    best_val_rank = float('inf') 
    best_atk_rank = float('inf') # [NEW] 最佳攻击排位

    # CSV Header
    if log_file:
        with open(log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'valid_loss', 'train_rank', 'valid_rank', 'attack_rank', 'best_loss', 'best_rank', 'best_attack'])

    if model_save_log:
        with open(model_save_log, mode='a') as f:
            f.write(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training started.\n")

    start_time = time.time()
    epoch = 0
    n = len(train_loader)

    while True:
        # --- Training Loop ---
        iter = -1
        model.train()
        train_loss = 0
        epoch_train_rank_sum = 0.0
        train_total_samples = 0

        for X_batch, Y_batch in train_loader:
            if device is not None:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            
            if not MLP: Y_pred = model(X_batch.unsqueeze(1))
            else: Y_pred = model(X_batch)
            
            loss = F.nll_loss(Y_pred, Y_batch)/math.log(2)

            # --- [关键修改 1] ---
            # 如果是 Adam，不需要 create_graph=True (这会浪费大量显存且变慢)
            # 只有 Scoop 需要它来计算 Hessian
            if use_hessian:
                loss.backward(create_graph=True)
            else:
                loss.backward() # Adam 默认走这里
            
            with torch.no_grad():
                true_vals = Y_pred.gather(1, Y_batch.view(-1, 1))
                ranks = (Y_pred > true_vals).sum(dim=1).float()
                epoch_train_rank_sum += ranks.sum().item()
                train_total_samples += X_batch.size(0)

            # --- [关键修改 2] ---
            # 安全调用 Hessian (防止 Adam 报错)
            if use_hessian and (iter % hessian_update == hessian_update - 1):
                optimizer.hutchinson_hessian()
                
            optimizer.step()
            train_loss += loss.item()
            iter += 1
            if verbose:
                print('Iteration ', iter, '/', n, '   Train Loss: ', train_loss/(iter+1), end='\r')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_train_rank = epoch_train_rank_sum / train_total_samples if train_total_samples > 0 else 255.0

        # --- Validation Loop ---
        model.eval()
        valid_loss = 0
        epoch_val_rank_sum = 0.0
        val_total_samples = 0
        
        # [NEW] 初始化全局 Key 概率累加器
        global_key_probs = torch.zeros(256, device=device)
        
        # 我们需要手动跟踪 batch index 以获取 plaintexts
        batch_idx = 0
        batch_size = valid_loader.batch_size

        with torch.no_grad():
            for X_batch, Y_batch in valid_loader:
                if device is not None:
                    X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                
                if not MLP: Y_pred = model(X_batch.unsqueeze(1))
                else: Y_pred = model(X_batch)
                
                loss = F.nll_loss(Y_pred, Y_batch)/math.log(2)
                valid_loss += loss.item()
                
                # 1. Mean Rank (Label Rank)
                true_vals = Y_pred.gather(1, Y_batch.view(-1, 1))
                ranks = (Y_pred > true_vals).sum(dim=1).float()
                epoch_val_rank_sum += ranks.sum().item()
                val_total_samples += X_batch.size(0)

                # 2. Attack Rank (Key Rank)
                if val_plaintexts is not None:
                    # 获取当前 Batch 对应的 Plaintexts
                    start_i = batch_idx * batch_size
                    end_i = start_i + X_batch.size(0)
                    current_pts = val_plaintexts[start_i:end_i].to(device)
                    
                    # 累加 Key 概率
                    batch_key_scores = compute_key_rank(Y_pred, current_pts, device)
                    global_key_probs += batch_key_scores
                
                batch_idx += 1
        
        avg_valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        avg_valid_rank = epoch_val_rank_sum / val_total_samples if val_total_samples > 0 else 255.0
        
        # 计算最终的 Key Rank
        current_attack_rank = 255.0
        if val_plaintexts is not None:
            # 假设目标 Key 是 0 (ASCAD 默认)
            # Rank = 有多少个 Key 的概率比 Key[0] 大
            target_score = global_key_probs[0] # Key 0
            current_attack_rank = (global_key_probs > target_score).sum().item()

        # --- 保存与日志 ---
        # 修复了上一轮的变量缺失问题
        current_epoch = epoch + 1
        current_time_str = datetime.datetime.now().strftime('%H:%M:%S')

        is_best_loss = False
        is_best_rank = False
        is_best_attack = False
        save_msg = []

        if save_best_model:
            # 1. Best Loss
            if avg_valid_loss < best_val_loss:
                best_val_loss = avg_valid_loss
                is_best_loss = True
                torch.save(model.state_dict(), best_path)
                save_msg.append(f"Best Loss ({best_val_loss:.4f})")
                if model_save_log:
                    with open(model_save_log, mode='a') as f:
                        f.write(f"[{current_time_str}] Epoch {current_epoch}: Updated {best_path} (Loss: {best_val_loss:.6f})\n")
            
            # 2. Best Mean Rank
            if avg_valid_rank < best_val_rank:
                best_val_rank = avg_valid_rank
                is_best_rank = True
                torch.save(model.state_dict(), best_rank_path)
                save_msg.append(f"Best Rank ({best_val_rank:.2f})")
                if model_save_log:
                    with open(model_save_log, mode='a') as f:
                        f.write(f"[{current_time_str}] Epoch {current_epoch}: Updated {best_rank_path} (Rank: {best_val_rank:.4f})\n")
            
            # 3. Best Attack Rank (Key Rank) - 只记录，不额外保存文件
            if current_attack_rank < best_atk_rank:
                best_atk_rank = current_attack_rank
                is_best_attack = True
                # save_msg.append(f"Best Atk ({best_atk_rank:.0f})")

        if verbose:
            save_str = f" -> Saved: {', '.join(save_msg)}" if save_msg else ""
            print(f'Epoch {current_epoch}/{n_epochs} | '
                  f'Loss: {avg_train_loss:.4f}/{avg_valid_loss:.4f} | '
                  f'Rank: {avg_train_rank:.2f}/{avg_valid_rank:.2f} | '
                  f'Attack: {current_attack_rank:.0f}' # 打印 Key Rank
                  f'{save_str}' 
                  ' | Time: {:.2f}s'.format((n_epochs - epoch - 1) * (time.time() - start_time) / (epoch + 1)))
        
        if log_file:
            with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_epoch, avg_train_loss, avg_valid_loss, avg_train_rank, avg_valid_rank, current_attack_rank, is_best_loss, is_best_rank, is_best_attack])

        epoch += 1
        
        if epoch >= n_epochs:
            if not finetuning: break
            else:
                if valid_losses[-1] > entropy: break
        if finetuning and valid_losses[-1] > entropy+0.1: break
        if epoch > 1 and math.isnan(valid_losses[-1]): break
    
    print(f"Training finished. Saving final model to {path}...")
    torch.save(model.state_dict(), path)
            
    return train_losses, valid_losses, path