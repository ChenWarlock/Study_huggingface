# 文本多类别

## 数据集
以2020语言与智能技术竞赛：事件抽取任务中的数据作为多分类标签的样例数据，借助多标签分类模型来解决。dd

## 使用方法
1. 下载数据集链接: 链接: https://pan.baidu.com/s/14c5RNiqD4WQh9BXZqcIIow 提取码: cs3s，解压并放入data文件夹。格式见下载数据集样式。
2. 下载预训练模型，并放入pre_trained_model内。本实验用的是 https://huggingface.co/hfl/chinese-bert-wwm-ext
3. 可以修改config.py里面的参数，具体见注释。
4. 配置好后直接python main.py即可
5. 由于gpu有限，暂时没有跑完，也没有跑出来精度多少。

ps：dev数据与test数据一样；初学者写的代码，质量一般emmmm,仅供学习参考
