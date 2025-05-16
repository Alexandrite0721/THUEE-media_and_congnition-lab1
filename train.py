# train_and_eval.py

import os
import torch
from torch.utils.data import DataLoader
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from loss import contrastive_loss
from data_loader import Flickr8kDataset
from utils import SimpleTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

#供同学们参考

def evaluate_top_k(img_encoder, txt_encoder, dataloader, device, topk=(1, 5, 10)):
    img_encoder.eval()
    txt_encoder.eval()

    all_image_embeds = []
    all_text_embeds = []

    with torch.no_grad():
        for images, captions_ids in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            captions_ids = captions_ids.to(device)

            image_embed = img_encoder(images)  # [1, dim]
            text_embed = txt_encoder(captions_ids)  # [1, dim]

            all_image_embeds.append(image_embed.cpu())
            all_text_embeds.append(text_embed.cpu())

    all_image_embeds = torch.cat(all_image_embeds, dim=0)  # [N, D]
    all_text_embeds = torch.cat(all_text_embeds, dim=0)    # [N, D]

    # 归一化
    all_image_embeds = F.normalize(all_image_embeds, dim=1)
    all_text_embeds = F.normalize(all_text_embeds, dim=1)

    # 文本 -> 图像检索
    sim_matrix = torch.matmul(all_text_embeds, all_image_embeds.T)  # [N, N]
    txt2img_ranks = torch.argsort(sim_matrix, dim=1, descending=True)

    # 图像 -> 文本检索
    sim_matrix_T = sim_matrix.T  # [N, N]
    img2txt_ranks = torch.argsort(sim_matrix_T, dim=1, descending=True)

    def recall_at_k(ranks, topk):
        recalls = []
        for k in topk:
            match = [i in ranks[i][:k] for i in range(len(ranks))]
            recalls.append(np.mean(match))
        return recalls

    r_txt2img = recall_at_k(txt2img_ranks, topk)
    r_img2txt = recall_at_k(img2txt_ranks, topk)

    print("\n📈 Text → Image Retrieval:")
    for i, k in enumerate(topk):
        print(f"Recall@{k}: {r_txt2img[i]*100:.2f}%")


    return r_txt2img, r_img2txt

# 实现 evaluate 函数
def evaluate(img_encoder, txt_encoder, dataloader, device):
    img_encoder.eval()
    txt_encoder.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, captions_ids in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            captions_ids = captions_ids.to(device)

            image_embeds = img_encoder(images)
            text_embeds = txt_encoder(captions_ids)
            loss = contrastive_loss(image_embeds, text_embeds)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main():
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 文件路径，根据实际调整
    token_file = "Flickr8k/captions.txt"         # 总的 captions 文件，用于构建词表
    train_token_file = "Flickr8k/train_captions.txt"  # 训练集，格式： image,caption
    val_token_file = "Flickr8k/val_captions.txt"      # 验证集
    test_token_file = "Flickr8k/test_captions.txt"    # 测试集
    
    # 读取所有 caption 用于构建总词表（假设以 tab 分隔，如果不是，请修改 split 参数）
    with open(token_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    captions = [line.strip().split(',')[1] for line in lines if line.strip()]
    
    # 构建统一的 tokenizer
    tokenizer = SimpleTokenizer(captions, min_freq=1)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # 构建数据集与 DataLoader：训练集、验证集、测试集
    train_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",      # 图片所在目录
        captions_file=train_token_file,   # 训练集 captions 文件，格式： image<TAB>caption
        tokenizer=tokenizer
    )
    val_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",
        captions_file=val_token_file,     # 验证集 captions 文件
        tokenizer=tokenizer
    )
    test_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",
        captions_file=test_token_file,    # 测试集 captions 文件
        tokenizer=tokenizer
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    # 为保证评估稳定，每个 batch 使用 batch_size=1
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    
    # 构造模型（设定 embed_dim=256）
    embed_dim = 256
    img_encoder = ImageEncoder(embed_dim=embed_dim).to(device)
    txt_encoder = TextEncoder(vocab_size, embed_dim=embed_dim).to(device)
    
    # 优化器
    optimizer = torch.optim.Adam(
        list(img_encoder.parameters()) + list(txt_encoder.parameters()),
        lr=1e-4
    )
    
    best_val_loss = float('inf')
    epochs = 40
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        img_encoder.train()
        txt_encoder.train()
        epoch_loss = 0.0
        
        for images, captions_ids in train_dataloader:
            images = images.to(device)
            captions_ids = captions_ids.to(device)
            
            image_embeds = img_encoder(images)      # [batch, embed_dim]
            text_embeds = txt_encoder(captions_ids)   # [batch, embed_dim]
            loss = contrastive_loss(image_embeds, text_embeds)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # 计算测试集损失
        test_loss = evaluate(img_encoder, txt_encoder, test_dataloader, device)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch+1}/{epochs}]: Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        # 如果验证集有改善，则保存最佳模型
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            checkpoint = {
                'epoch': epoch + 1,
                'img_encoder_state_dict': img_encoder.state_dict(),
                'txt_encoder_state_dict': txt_encoder.state_dict(),
                'tokenizer_vocab': tokenizer.word2idx,
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, "best_clip_model.pth")
            print(f"    > Best model updated at epoch {epoch+1} ")
    
    # 训练完成，最终在测试集上评估
    final_test_loss = evaluate(img_encoder, txt_encoder, test_dataloader, device)
    print(f"Final Test Loss: {final_test_loss:.4f}")

    # 绘制训练和测试损失在同一幅图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.savefig('train_and_test_loss.png')
    plt.close()

    # 绘制训练损失图
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.savefig('train_loss.png')
    plt.close()

    # 绘制测试损失图
    plt.figure(figsize=(10, 5))
    plt.plot(test_losses, label='Test Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.legend()
    plt.savefig('test_loss.png')
    plt.close()

if __name__ == "__main__":
    main()
