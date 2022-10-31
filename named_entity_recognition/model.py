import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchcrf import CRF
from torch import tensor
import torch


class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, model_url: str, num_classes: int, rnn_dim: int = 128):
        super(BERT_BiLSTM_CRF, self).__init__()
        ## 导入预训练模型与配置文件
        self.config = AutoConfig.from_pretrained(model_url)
        self.plm = AutoModel.from_pretrained(model_url)
        ## 定义一层dropout
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        ## 定义bilstm层
        self.birnn = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=rnn_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        ## 定义全连接层，进行投影
        self.hidden2tag = nn.Linear(in_features=rnn_dim * 2, out_features=num_classes)
        ## 定义CRF层
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(
        self, input_ids: tensor, attention_mask: tensor, tags: tensor
    ) -> tensor:
        ## 获取bert输出向量
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        ## 取出最后一层的向量
        sequence_output = outputs[0]
        ## 送入bilstm中
        sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        ## 使用crf计算损失
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def predict(self, input_ids: tensor, attention_mask: tensor) -> list:
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, attention_mask.byte())
