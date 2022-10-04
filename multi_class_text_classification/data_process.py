import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def get_data(data_url: str) -> dict:
    train_data = []
    dev_data = []
    test_data = []
    labels = []
    with open(os.path.join(data_url, 'train.txt'), 'r', encoding='utf8') as f:
        for line in f:
            data, label = line.strip().split('\t')
            train_data.append((data, label))
    with open(os.path.join(data_url, 'dev.txt'), 'r', encoding='utf8') as f:
        for line in f:
            data, label = line.strip().split('\t')
            dev_data.append((data, label))
    with open(os.path.join(data_url, 'test.txt'), 'r', encoding='utf8') as f:
        for line in f:
            data, label = line.strip().split('\t')
            test_data.append((data, label))
    with open(os.path.join(data_url, 'labels.txt'), 'r', encoding='utf8') as f:
        for line in f:
            label = line.strip()
            labels.append(label)
    return {'train': train_data, 'dev': dev_data, 'test': test_data, 'label': labels}


class MulticlassDataset(Dataset):
    def __init__(self, data: list, model_url: str, max_len: int):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        text = str(self.data[item][0])
        label = int(self.data[item][1])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text,
        }


def create_data_loader(
    data_url: str, model_url: str, max_len: int, batch_size: int
) -> DataLoader:
    data = get_data(data_url)
    train_data = data['train']
    dev_data = data['dev']
    test_data = data['test']
    labels = data['label']

    dataset_train = MulticlassDataset(
        data=train_data,
        model_url=model_url,
        max_len=max_len,
    )
    dataset_dev = MulticlassDataset(
        data=dev_data,
        model_url=model_url,
        max_len=max_len,
    )
    dataset_test = MulticlassDataset(
        data=test_data,
        model_url=model_url,
        max_len=max_len,
    )

    train_dataLoader = DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True
    )
    dev_dataLoader = DataLoader(
        dataset=dataset_dev, batch_size=batch_size, shuffle=False
    )
    test_dataLoader = DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=False
    )
    return train_dataLoader, dev_dataLoader, test_dataLoader, labels
