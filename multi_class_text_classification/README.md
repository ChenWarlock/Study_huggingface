# 文本多类别

## 数据集
IFLYTEK' 长文本分类 Long Text classification，https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip

## 使用方法
1. 下载数据集链接: https://pan.baidu.com/s/16kwZcjM-1B7k2vQzoVomDA 提取码: xht8，解压并放入data文件夹。格式见下载数据集样式。
2. 下载预训练模型，并放入pre_trained_model内。本实验用的是 https://huggingface.co/hfl/chinese-bert-wwm-ext
3. 可以修改config.py里面的参数，具体见注释。
4. 配置好后直接python main.py即可
5. 由于gpu有限，暂时没有跑完，也没有跑出来精度多少。

ps：初学者写的代码，质量一般emmmm,仅供学习参考
