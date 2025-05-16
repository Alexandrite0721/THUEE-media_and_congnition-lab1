import torch
import torch.nn.functional as F
#请在这里写出对比损失函数
def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    """
    image_embeds, text_embeds: [batch, embed_dim]
    """
    # 归一化嵌入向量以计算余弦相似度
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # 计算图像和文本嵌入的相似度矩阵
    logits = torch.matmul(image_embeds, text_embeds.T) / temperature  # [batch, batch]

    # 计算标签，正样本为对角线上的元素
    batch_size = image_embeds.size(0)
    labels = torch.arange(batch_size, device=image_embeds.device)

    # 计算图像到文本的 InfoNCE 损失
    loss_image_to_text = F.cross_entropy(logits, labels)

    # 计算文本到图像的 InfoNCE 损失
    loss_text_to_image = F.cross_entropy(logits.T, labels)

    # 总损失为两者的平均
    loss = (loss_image_to_text + loss_text_to_image) / 2
    return loss