import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import re
import numpy as np


def read_data(file_path: str) -> tuple:
    token_docs = []
    tag_docs = []
    with open(file_path, 'r', encoding='utf8') as f:
        raw_text = f.read().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            if len(line.split(' ')) == 2:
                token, tag = line.split(' ')
            else:
                continue
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)
    return token_docs, tag_docs


def encode_tags(tags: list, encodings: dict, max_len: int, tag2id: dict) -> list:
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # 创建全由O组成的矩阵
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * tag2id['O']
        arr_offset = np.array(doc_offset)

        if len(doc_labels) >= max_len - 2:  # 防止异常
            doc_labels = doc_labels[: max_len - 2]
        doc_enc_labels[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
    return encoded_labels


class NerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_data_loader(
    data_url: str, model_url: str, max_len: int, batch_size: int
) -> DataLoader:
    ## 读入数据
    train_texts, train_tags = read_data(os.path.join(data_url, 'train.txt'))
    dev_texts, dev_tags = read_data(os.path.join(data_url, 'dev.txt'))
    test_texts, test_tags = read_data(os.path.join(data_url, 'test.txt'))

    ## 抽取类别
    unique_tags = ['B-PER', 'I-PER','B-ORG','I-ORG','B-LOC','I-LOC','O']
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    ## 分词
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    train_encodings = tokenizer(
        train_texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
    )
    dev_encodings = tokenizer(
        dev_texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
    )
    test_encodings = tokenizer(
        test_texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding='max_length',
        truncation=True,
        max_length=max_len,
    )

    ## 将特殊词的类别填补为"O"，计算损失时可以忽略
    train_labels = encode_tags(train_tags, train_encodings, max_len, tag2id)
    dev_labels = encode_tags(dev_tags, dev_encodings, max_len, tag2id)
    test_labels = encode_tags(test_tags, test_encodings, max_len, tag2id)

    ## 删除多余的元素
    train_encodings.pop("offset_mapping")
    dev_encodings.pop("offset_mapping")
    test_encodings.pop("offset_mapping")

    ## 制作Dataset
    train_dataset = NerDataset(train_encodings, train_labels)
    dev_dataset = NerDataset(dev_encodings, dev_labels)
    test_dataset = NerDataset(test_encodings, test_labels)

    ## 制作成DataLoader
    train_dataLoader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    dev_dataLoader = DataLoader(
        dataset=dev_dataset, batch_size=batch_size, shuffle=False
    )
    test_dataLoader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_dataLoader, dev_dataLoader, test_dataLoader, unique_tags
