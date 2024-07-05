# UUD-Fusion
This is the code for article An Unsupervised Universal Image Fusion Approach via Generative Diffusion
Model. The detailed code will be uploaded later.

这是文章 An Unsupervised Universal Image Fusion Approach via Generative Diffusion
Model 的代码。 详细代码稍后会上传。

# 修改日期2024.07.05 Date of revision 2024.07.05
使用的数据集和训练的参数文件稍后会上传

# 数据集 Datasets
## 训练数据集 Training Datasets
link 链接：https://pan.baidu.com/s/1up-FGeMLt0_LQACDWjDYoQ?pwd=6356 extraction code 提取码：6356

## 测试数据集 Test Datasets
link 链接：https://pan.baidu.com/s/1cP7DKddQHjxO6W6ABZc7cg?pwd=hwso extraction code 提取码：hwso
or
https://drive.google.com/file/d/1-5-rZEbpCgVHfy-ki0m-bXdqT7_Sz6Qq/view?usp=drivesdk

# 使用方法 How to use
## 训练方法，需要根据实际情况修改train文件中数据集地址和训练参数的保存地址，数据集读取可以自行根据实际情况修改，此处提供了一种参考
Training method: you need to modify the dataset address in the TRAIN file and the save address of the training parameters according to the actual situation. The dataset reading can be modified by yourself according to the actual situation. A reference is provided here.

首先，生成预训练模型的训练参数文件，运行pretrain19-3.py，需要自行修改训练数据集和参数文件保存位置

然后训练正式阶段的扩散模型，不同任务的训练方法如下：
多聚焦图像融合任务的训练，运行train39.py

多曝光图像融合任务的训练，运行train39.py

红外和可见光图像融合任务的训练，运行train39-2.py

医学图像融合任务的训练，运行train39-3.py

Firstly, generate the training parameter file for the pre-trained model, run pretrain19-3.py, you need to modify the training dataset and parameter file save location by yourself

Then train the diffusion model in the formal stage, the training methods for different tasks are as follows:
Training for multi-focus image fusion task, run train39.py

Training for the multi-exposure image fusion task, running train39.py

Training for infrared and visible image fusion task, running train39-2.py

Training for medical image fusion task, running train39-3.py

## 测试方法，需要根据实际情况修改test文件中训练参数的读取地址和融合图像保存地址
Test method: you need to modify the reading address of the training parameters and the fusion image saving address in the TEST file according to the actual situation.

多聚焦图像融合任务测试，运行test39.py

多曝光图像融合任务测试，运行test39-6.py

红外和可见光图像融合任务测试，运行test39-17.py

医学图像融合任务测试，运行test39-8.py

Multi-focus image fusion task test, running test39.py

Multi-exposure image fusion task test, running test39-6.py

Infrared and visible image fusion task test, running test39-17.py

Medical image fusion task test, run test39-8.py

## 关于测试实验结果 About the test experiment results
results文件夹中给出了使用不同数据集、多种图像融合任务的实验结果

Experimental results using different datasets and multiple image fusion tasks are given in the results folder.

## 保存的训练文件下载地址
Download address for saved training files

稍后上传

Uploading later

# 关于引用 About the citation
文章尚在审稿，稍后会发布

The article is still under review and will be posted later.
