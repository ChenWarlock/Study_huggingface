import os

import numpy as np
import torch
from sklearn import metrics
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_scheduler

from model import BERT_BiLSTM_CRF


class Trainer(object):
    def __init__(
        self,
        train_dataloader: DataLoader,  # 训练的dataloader
        dev_dataloader: DataLoader,  # 验证的dataloader
        test_dataloader: DataLoader,  # 预测的dataloader
        labels: list,  # 类别数
        config: object,  # 一些超参数的配置
    ) -> None:
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.labels = labels
        self.config = config

    def get_group_parameters(self, model: nn.Module) -> list:
        params = list(model.named_parameters())
        no_decay = ['bias,', 'LayerNorm']
        other = ['dense', 'fc']
        no_main = no_decay + other
        param_group = [
            {
                'params': [p for n, p in params if not any(nd in n for nd in no_main)],
                'weight_decay': 1e-2,
            },
            {
                'params': [
                    p
                    for n, p in params
                    if not any(nd in n for nd in other)
                    and any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0,
            },
            {
                'params': [
                    p
                    for n, p in params
                    if any(nd in n for nd in other) and any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0,
            },
            {
                'params': [
                    p
                    for n, p in params
                    if any(nd in n for nd in other)
                    and not any(nd in n for nd in no_decay)
                ],
                'weight_decay': 1e-2,
            },
        ]
        return param_group

    def train(self) -> None:
        ## 总的训练次数
        step_total = (
            self.config.epochs * len(self.train_dataloader) * self.config.batch_size
        )
        ## warm up的次数
        warmup_steps = int(step_total * self.config.num_warmup_rate)

        ## 定义模型与损失函数，这样定义是为了再验证时方便调用
        self.model = BERT_BiLSTM_CRF(
            self.config.model_url, len(self.labels), self.config.rnn_dim
        )
        self.model.to(self.config.device)

        ## 定义优化器和学习了变化曲线，能够随着训练变化学习率大小
        optimizer = AdamW(self.get_group_parameters(self.model), lr=self.config.lr)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=step_total,
        )

        ## 打印一些基本的参数
        print(
            "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 基本参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        print(f"Num Epochs = {self.config.epochs}")
        print(f"Batch size per GPU = {self.config.batch_size}")
        print(f"Total step = {step_total}")
        print(f"Warm up step = {warmup_steps}")
        print(f"Accumulation step = {self.config.accumulation_steps}")
        print(f"Wheather autocast = {self.config.is_autocast}")
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 基本参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
        )

        ## 打印模型结构
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 模型结构 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        for name, parameters in self.model.named_parameters():
            print(name, ':', parameters.size())
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 模型结构 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
        )

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 开始训练 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        f1_best = 0
        for epoch in range(self.config.epochs):
            ## 定义循环进度条,长度ncols可以自行修改
            loop = tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                ncols=100,
                ascii=True,
            )
            for i, batch in loop:
                self.model.train()
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                target = batch['labels'].to(self.config.device)

                ## 是否启用混合精度训练
                if self.config.is_autocast:
                    with autocast():
                        loss = self.model(input_ids, attention_mask, target)
                else:
                    loss = self.model(input_ids, attention_mask, target)
                loss.backward()

                ## 梯度累加步数
                if (i + 1) % self.config.accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                ## 进度条显示更新
                loop.set_description(f'Epoch [{epoch+1}/{self.config.epochs}]')
                loop.set_postfix(loss=loss.item())

            ## 是否验证模型效果
            if self.config.is_eval and (epoch + 1) % self.config.eval_epoch == 0:
                f1 = self.eval()
                ## 是否保存最好模型
                if self.config.save_best:
                    if not os.path.exists(self.config.save_dir):
                        os.makedirs(self.config.save_dir)
                    if f1 > f1_best:
                        f1_best = f1
                        print(
                            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 保存模型 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
                        )
                        print(f"Save Path = {self.config.save_dir}")
                        print(f"Best F1 = {f1_best:.4f}")
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.config.save_dir, 'best_model.pth'),
                        )
                        print(
                            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 保存成功 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
                        )

        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 训练完成 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
        )

    def eval(self) -> float:
        print(
            "\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 开始验证 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        )
        self.model.eval()
        all_pre = []
        all_true = []
        total_loss = 0
        loop = tqdm(
            enumerate(self.dev_dataloader),
            total=len(self.dev_dataloader),
            ncols=100,
            ascii=True,
        )
        with torch.no_grad():
            for i, batch in loop:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                target = batch['labels'].to(self.config.device)
                loss = self.model(input_ids, attention_mask, target)
                total_loss += loss.item()

                predict = self.model.predict(input_ids, attention_mask)
                masks = torch.sum(batch['attention_mask'], dim=1)
                labels = [
                    label[1 : mask - 1].tolist()
                    for mask, label in zip(masks, target.cpu())
                ]
                predict = [i[1:-1] for i in predict]
                all_pre.extend(predict)
                all_true.extend(labels)
                loop.set_description('Evaling')
                loop.set_postfix(loss=loss.item())

        func = lambda x: [y for l in x for y in func(l)] if type(x) is list else [x]
        all_true = func(all_true)
        all_pre = func(all_pre)
        acc = metrics.accuracy_score(all_true, all_pre)
        f1_mi = metrics.f1_score(all_true, all_pre, average='micro')
        f1_ma = metrics.f1_score(all_true, all_pre, average='macro')
        print(
            f'>> 验证结果 >>:  Loss:{total_loss:.4f}  Acc:{acc:.4f}  MicroF1:{f1_mi:.4f}  MacroF1:{f1_ma:.4f}'
        )
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 验证完成 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
        )
        return f1_mi

    def predict(self) -> None:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 开始预测 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.model.load_state_dict(
            torch.load(os.path.join(self.config.save_dir, 'best_model.pth'))
        )
        self.model.eval()
        all_pre = []
        all_true = []
        total_loss = 0
        loop = tqdm(
            enumerate(self.test_dataloader),
            total=len(self.test_dataloader),
            ncols=100,
            ascii=True,
        )
        with torch.no_grad():
            for i, batch in loop:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                target = batch['labels'].to(self.config.device)
                loss = self.model(input_ids, attention_mask, target)
                total_loss += loss.item()

                predict = self.model.predict(input_ids, attention_mask)
                masks = torch.sum(batch['attention_mask'], dim=1)
                labels = [
                    label[1 : mask - 1].tolist()
                    for mask, label in zip(masks, target.cpu())
                ]
                predict = [i[1:-1] for i in predict]
                all_pre.extend(predict)
                all_true.extend(labels)
                loop.set_description('Predicting')
                loop.set_postfix(loss=loss.item())

        func = lambda x: [y for l in x for y in func(l)] if type(x) is list else [x]
        true = func(all_true)
        pre = func(all_pre)
        acc = metrics.accuracy_score(true, pre)
        f1_mi = metrics.f1_score(true, pre, average='micro')
        f1_ma = metrics.f1_score(true, pre, average='macro')
        print(
            f'>> 预测结果 >>:  Loss:{total_loss:.4f}  Acc:{acc:.4f}  MicroF1:{f1_mi:.4f}  MacroF1:{f1_ma:.4f}'
        )
        for i in range(len(all_pre)):
            for j in range(len(all_pre[i])):
                all_pre[i][j] = self.labels[all_pre[i][j]]

        with open('predict_result.txt', 'a', encoding='utf8') as f:
            for item in all_pre:
                f.write('\t'.join(item) + '\n')
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 预测结束 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
        )
