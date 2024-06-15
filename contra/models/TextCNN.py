import torch.nn as nn
import torch
from setting import DEVICE
from tokenizer import MyTokenizer

class TextCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, maps, embedding_path) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        self.article_class_num = len(maps["article2idx"])
        self.hid_dim = hid_dim

        kernels = (2, 3, 4)
        self.tokenizer = MyTokenizer(embedding_path)
        # vectors = self.tokenizer.load_embedding()
        # vectors = torch.Tensor(vectors)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # self.embedding.weight.data.copy_(vectors)

        self.convs = nn.ModuleList(
            [nn.Conv1d(self.emb_dim, self.hid_dim,  kernel_size=i) for i in kernels])

        self.fc1 = nn.Linear(len(kernels) * self.hid_dim, self.hid_dim)
        self.fc_article = nn.Linear(self.hid_dim, self.article_class_num)
        self.fc_charge = nn.Linear(self.hid_dim, self.charge_class_num)
        # self.fc_judge = nn.Linear(self.hid_dim, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, data):
        text = data["justice"]["input_ids"].to(DEVICE)
        x = self.embedding(text)

        def conv_and_pool(x, conv):
            x = x.permute(0, 2, 1)
            x = torch.nn.ReLU()(conv(x))
            x = torch.nn.MaxPool1d(x.shape[-1])(x)
            return x
        res = [conv_and_pool(x, conv) for conv in self.convs]
        # 255, 170, 127
        tot = torch.cat(res, dim=2)
        out = tot.flatten(start_dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = torch.nn.ReLU()(out)

        out_charge = self.fc_charge(out)
        out_article = self.fc_article(out)
        # out_judge = self.fc_judge(out)
        return {
            "article": out_article,
            "charge": out_charge,
            # "judge": out_judge
        }
