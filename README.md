# UUD-Fusion
This is the code for article An Unsupervised Universal Image Fusion Approach via Generative Diffusion
Model. The detailed code will be uploaded later.

这是文章 An Unsupervised Universal Image Fusion Approach via Generative Diffusion
Model 的代码。 详细代码稍后会上传。

## 修改日期2024.05.11
使用的数据集和训练的参数文件稍后会上传

## 训练方法，需要根据实际情况修改train文件中数据集地址和训练参数的保存地址，数据集读取可以自行根据实际情况修改，此处提供了一种参考
首先，生成预训练模型的训练参数文件，运行pretrain19-3.py，需要自行修改训练数据集和参数文件保存位置

然后训练正式阶段的扩散模型，不同任务的训练方法如下：
多聚焦图像融合任务的训练，运行train39.py

多曝光图像融合任务的训练，运行train39.py

红外和可见光图像融合任务的训练，运行train39-2.py

医学图像融合任务的训练，运行train39-3.py

## 测试方法，需要根据实际情况修改test文件中训练参数的读取地址和融合图像保存地址
多聚焦图像融合任务测试，运行test39.py

多曝光图像融合任务测试，运行test39-6.py

红外和可见光图像融合任务测试，运行test39-17.py

医学图像融合任务测试，运行test39-8.py

## 关于测试实验结果
results文件夹中给出了使用不同数据集，多种图像融合任务的实验结果

## 关于使用的训练数据集和训练的参数文件
稍后上传

# 关于引用
文章尚在审稿，稍后会发布
