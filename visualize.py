import torch
from torch.utils.data import DataLoader
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from data_loader import Flickr8kDataset
from utils import SimpleTokenizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

# 加载模型和数据
def load_model_and_data(checkpoint_path, token_file, train_token_file, val_token_file, test_token_file):
    # 加载检查点
    checkpoint = torch.load(checkpoint_path)

    # 读取所有 caption 用于构建总词表
    with open(token_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    captions = [line.strip().split(',')[1] for line in lines if line.strip()]

    # 构建统一的 tokenizer
    tokenizer = SimpleTokenizer(captions, min_freq=1)
    vocab_size = len(tokenizer)

    # 构建数据集与 DataLoader：测试集
    test_dataset = Flickr8kDataset(
        root_dir="Flickr8k/images",
        captions_file=test_token_file,
        tokenizer=tokenizer
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    # 构造模型（设定 embed_dim=256）
    embed_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_encoder = ImageEncoder(embed_dim=embed_dim).to(device)
    txt_encoder = TextEncoder(vocab_size, embed_dim=embed_dim).to(device)

    # 加载模型状态字典
    img_encoder.load_state_dict(checkpoint['img_encoder_state_dict'])
    txt_encoder.load_state_dict(checkpoint['txt_encoder_state_dict'])

    return img_encoder, txt_encoder, test_dataloader, tokenizer, device

# 计算 Top-K 召回率
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

    def recall_at_k(ranks, topk):
        recalls = []
        for k in topk:
            match = [i in ranks[i][:k] for i in range(len(ranks))]
            recalls.append(np.mean(match))
        return recalls

    r_txt2img = recall_at_k(txt2img_ranks, topk)

    return r_txt2img

# 文本 -> 图像 Top-K 检索示例的可视化
def visualize_text_to_image_retrieval(img_encoder, txt_encoder, test_dataloader, tokenizer, device, query_caption, top_k=5):
    img_encoder.eval()
    txt_encoder.eval()

    all_image_embeds = []
    all_image_paths = []

    with torch.no_grad():
        for images, captions_ids, image_paths in tqdm(test_dataloader, desc="Extracting image embeddings"):
            images = images.to(device)
            image_embed = img_encoder(images)  # [1, dim]
            all_image_embeds.append(image_embed.cpu())
            all_image_paths.extend(image_paths)

    all_image_embeds = torch.cat(all_image_embeds, dim=0)  # [N, D]
    all_image_embeds = F.normalize(all_image_embeds, dim=1)

    # 对查询文本进行编码
    query_tokens = tokenizer.encode(query_caption)
    query_tensor = torch.tensor(query_tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embed = txt_encoder(query_tensor)
        query_embed = F.normalize(query_embed, dim=1).cpu()

    # 计算相似度
    sim_scores = torch.matmul(query_embed, all_image_embeds.T).squeeze()
    top_k_indices = torch.argsort(sim_scores, descending=True)[:top_k]

    # 可视化结果
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(top_k_indices):
        img_path = all_image_paths[idx]
        img = plt.imread(img_path)
        plt.subplot(1, top_k, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Top-{i + 1}")

    plt.suptitle(f"Query: {query_caption}")
    plt.show()

def main():
    checkpoint_path = "best_clip_model.pth"
    token_file = "Flickr8k/captions.txt"
    train_token_file = "Flickr8k/train_captions.txt"
    val_token_file = "Flickr8k/val_captions.txt"
    test_token_file = "Flickr8k/test_captions.txt"

    img_encoder, txt_encoder, test_dataloader, tokenizer, device = load_model_and_data(
        checkpoint_path, token_file, train_token_file, val_token_file, test_token_file
    )

    # 计算 Top-K 召回率
    topk = (1, 5, 10)
    r_txt2img = evaluate_top_k(img_encoder, txt_encoder, test_dataloader, device, topk)

    # 列出模型训练结束后的性能指标
    print("\n📈 Text → Image Retrieval:")
    for i, k in enumerate(topk):
        print(f"Recall@{k}: {r_txt2img[i]*100:.2f}%")

    # 文本 -> 图像 Top-K 检索示例的可视化
    query_caption = "A little girl covered in paint"
    visualize_text_to_image_retrieval(img_encoder, txt_encoder, test_dataloader, tokenizer, device, query_caption, top_k=5)

if __name__ == "__main__":
    main()