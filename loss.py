import torch
import torch.nn.functional as F
#请在这里写出对比损失函数
def contrastive_loss(image_embeds, text_embeds, temperature=0.07):
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = torch.matmul(image_embeds, text_embeds.T) / temperature
    labels = torch.arange(len(image_embeds), device=image_embeds.device)

    loss_image_text = F.cross_entropy(logits, labels)
    loss_text_image = F.cross_entropy(logits.T, labels)

    return (loss_image_text + loss_text_image) / 2