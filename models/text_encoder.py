import torch.nn as nn
from .lstm_custom import LSTMTextEncoder
from .transformer_encoder import TransformerTextEncoder

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, encoder_type='lstm'):
        """
        参数说明：
          - vocab_size: 词表大小
          - embed_dim: 嵌入维度（同时也是目标输出维度）
          - hidden_dim: LSTM 隐藏层维度（仅在 encoder_type='lstm' 时有效）
          - encoder_type: 指定使用哪种编码器，可选 'lstm' 或 'transformer'
        """
        super().__init__()
        
        if encoder_type == 'lstm':
            # 使用手写版 LSTMTextEncoder，注意此处 LSTMTextEncoder 内部已包含词嵌入层和全连接映射
            self.encoder = LSTMTextEncoder(vocab_size, embed_dim, hidden_dim)
        elif encoder_type == 'transformer':
            print("true")
            # 使用手写版 TransformerTextEncoder，注意内部参数可根据需求调整，如头数、最大序列长度等
            self.encoder = TransformerTextEncoder(vocab_size, embed_dim, num_heads=4, padding_idx=0, max_len=100)
        else:
            raise ValueError("Unsupported encoder_type: choose either 'lstm' or 'transformer'")
    
    def forward(self, captions):
        # 直接调用内部 encoder 模块进行前向传播
        return self.encoder(captions)  # 输出形状：[batch, embed_dim]

