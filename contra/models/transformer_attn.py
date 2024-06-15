import torch.nn as nn
import torch
from setting import DEVICE


class transformer_attn(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, maps) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        self.article_class_num = len(maps["article2idx"])

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.transformer = nn.TransformerEncoderLayer(d_model=emb_dim, dim_feedforward=hid_dim, nhead=4,
                            batch_first=True, dropout=0.5)

        self.w = nn.Parameter(torch.zeros(hid_dim))
        self.fc_input_dim = hid_dim
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.fc_input_dim, self.hid_dim)
        self.fc_article = nn.Linear(self.hid_dim, self.article_class_num)
        self.fc_charge = nn.Linear(self.hid_dim, self.charge_class_num)
        # self.fc_k = nn.Linear(self.hid_dim, 1)

    def forward(self, data):
        text = data["justice"]["input_ids"].to(DEVICE)
        # text = data["rationale"].to(DEVICE)
        x = self.embedding(text)
        hidden = self.transformer(x)  # [64, 512, 256]

        mat = nn.Tanh()(hidden)  # [64, 512, 256]
        alpha = nn.Softmax(dim=1)(torch.matmul(mat, self.w)
                                  ).unsqueeze(-1)  # [64, 512, 1]
        out = hidden * alpha  # [64, 512, 256]
        # out = hidden
        out = torch.sum(out, dim=1)  # [64, 256]
        out = self.fc1(out)
        out = nn.ReLU()(out)
        out_charge = self.fc_charge(out)
        out_article = self.fc_article(out)
        # out_k = self.fc_k(out)
        return {
            "article": out_article,
            "charge": out_charge
            # "k": out_k
        }
