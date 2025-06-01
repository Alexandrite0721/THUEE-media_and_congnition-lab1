import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * ( -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):

        return x + self.pe[:, :x.size(1)].to(x.device)

# ----- 填空题 1: 实现 MultiHeadSelfAttention 模块 -----
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能整除 num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, mask=None):
        # x: [B, T, D]，D=embed_dim
        B, T, D = x.size()

        # 计算 Q, K, V 矩阵
        qkv = self.qkv_proj(x)  # [B, T, 3 * D]
        
        # 改变维度顺序，将batch放在前面
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv  # 每个形状为 [B, num_heads, T, head_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, T, T]
        
        if mask is not None:
            # 调整mask形状以匹配scores
            # mask形状: [B, 1, 1, T] -> 需要广播到 [B, num_heads, T, T]
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)  # [B, num_heads, T, T]
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, T, head_dim]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, T, D)  # [B, T, D]
        
        # 通过输出投影层
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        # 初始化前馈网络，先 Linear(embed_dim, embed_dim*4) -> ReLU -> Linear(embed_dim*4, embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        # 残差连接：先 LayerNorm，再 MultiHeadSelfAttention，再加上输入
        # 计算自注意力块输出，并加上原输入，实现残差连接
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        # 同理，对于前馈网络：先 LayerNorm，再前馈计算，再加上输入
        # 计算前馈网络输出，并加上残差
        x = x + self.ff(self.ln2(x))
        return x

# ----- 填空题 2: 实现 TransformerTextEncoder 模块 -----
class TransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, padding_idx=0, max_len=100):
        super().__init__()
        #初始化嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # 初始化位置编码器，参数为 embed_dim 和 max_len
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_len)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(embed_dim, embed_dim)
    #请完成forward函数   
    def forward(self, captions):
        # captions: [B, T]
        x = self.embedding(captions)  # [B, T, embed_dim]
        x = self.pos_encoder(x)       # [B, T, embed_dim]

        # 构造padding mask: [B, 1, 1, T]，1表示有效，0表示padding
        mask = (captions != self.embedding.padding_idx).unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        for layer in self.layers:
            x = layer(x, mask=mask)
        # 只对有效token做平均池化
        mask_float = (captions != self.embedding.padding_idx).float()  # [B, T]
        x = (x * mask_float.unsqueeze(-1)).sum(dim=1) / mask_float.sum(dim=1, keepdim=True)  # [B, embed_dim]
        out = self.fc(x)
        return F.normalize(out, p=2, dim=1)

# ----- 测试单元 -----
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 1000
    # 随机生成输入：形状为 [batch=4, T=20]
    dummy_captions = torch.randint(0, vocab_size, (4, 20)).to(device)
    # 实例化 TransformerTextEncoder，要求使用 4 层（num_layers=4）
    model = TransformerTextEncoder(vocab_size, embed_dim=256, num_heads=4, num_layers=4, padding_idx=0, max_len=100).to(device)
    output = model(dummy_captions)
    print("输出嵌入张量的形状：", output.shape)  # 期望输出：[4, 256]
