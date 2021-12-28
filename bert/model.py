from transformers import AutoModel
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, model_name, num_cls):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.logit_layer = nn.Linear(self.model.config.hidden_size, num_cls)

    def forward(self, inp):
        # tokens: [B, L], mask: [B, L]
        x = self.model(inp['input_tokens'],
                       inp['input_attn_mask'])[0]  # [B, L, D]
        cls_emb = x[:, 0, :]  # [B, D]
        logit = self.logit_layer(cls_emb)  # [B, C]
        return logit
