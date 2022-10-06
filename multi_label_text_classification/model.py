import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch import tensor


class Classifier(nn.Module):
    def __init__(self, model_url: str, num_classes: int, max_len: int) -> None:
        super(Classifier, self).__init__()
        ## 导入预训练模型与配置文件
        self.config = AutoConfig.from_pretrained(model_url)
        self.plm = AutoModel.from_pretrained(model_url)
        ## 获取模型隐藏层大小和类别数目
        self.hidden_size = self.config.hidden_size
        self.num_classes = num_classes
        ## 定义两个层，目的是获取句子最大的embedding和平均embedding
        self.maxpool = nn.MaxPool1d(max_len)
        self.avgpool = nn.AvgPool1d(max_len)
        ## 定义两个线性层与激活函数
        self.dense = nn.Linear(3 * self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        self.tanh = nn.Tanh()

    def forward(self, input_ids: tensor, attention_mask: tensor) -> None:
        ## 获取plm模型的输出结果
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        ## 获取cls、句子最大向量和句子平均向量
        sentence_emb = outputs.last_hidden_state[:, 0, :]
        output = outputs.last_hidden_state.permute(0, 2, 1).contiguous()
        maxpool_out = self.maxpool(output).squeeze(2)
        avgpool_out = self.avgpool(output).squeeze(2)
        ## 将三者拼接起来并经过两次线性层
        output = torch.cat((sentence_emb, maxpool_out, avgpool_out), 1)
        output = self.tanh(self.dense(output))
        logits = self.fc(output)
        return logits
