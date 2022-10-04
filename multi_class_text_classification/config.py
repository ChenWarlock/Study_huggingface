import torch


class Config(object):

    ## 模型数据路径配置
    model_url: str = './pre_trained_model/chinese-bert-wwm-ext'
    data_url: str = './data'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    ## 模型超参数配置
    lr: int = 1e-5
    max_len: int = 8  # 句子最大长度
    batch_size: int = 256  # 批量大小
    epochs: int = 5  # 训练轮数
    is_eval: bool = True  # 是否进行验证
    eval_epoch: int = 2  # 每隔多少轮进行验证
    save_best: bool = True  # 是否保存最好的模型
    save_dir: str = './best_checkpoint'  # 保存模型的路径
    accumulation_steps: int = 4  # 梯度累加
    is_autocast: bool = True  # 混合精度训练
    num_warmup_rate: int = 0.2  # warm_up步数
    is_predict: bool = True  # 是否进行模型预测
