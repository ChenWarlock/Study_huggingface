from data_process import create_data_loader
from config import Config
from trainer import Trainer
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    ## 定于全局的配置文件
    config = Config()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 加载数据集 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    ## 导入训练和验证数据集
    train_dataLoader, dev_dataLoader, test_dataLoader, unique_tags  = create_data_loader(
        config.data_url, config.model_url, config.max_len, config.batch_size
    )
    
    ## 定义模型训练类，并进行模型训练
    trainer_model = Trainer(
        train_dataLoader, dev_dataLoader, test_dataLoader, unique_tags, config
    )
    trainer_model.train()
    ## 读取最好的模型，并进行预测
    if config.is_predict:
        trainer_model.predict()
