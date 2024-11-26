# UUD-Fusion
This is the code for article  UUD-Fusion: An unsupervised universal image fusion approach via generative diffusion model. The detailed code has been uploaded.

这是文章 UUD-Fusion: An unsupervised universal image fusion approach via generative diffusion model 的代码。详细代码已经上传。

# 修改日期2024.11.26 Date of revision 2024.11.26

# 数据集 Datasets
## 训练数据集 Training Datasets
link 链接：https://pan.baidu.com/s/1up-FGeMLt0_LQACDWjDYoQ?pwd=6356 

extraction code 提取码：6356

## 测试数据集 Test Datasets
link 链接：https://pan.baidu.com/s/1cP7DKddQHjxO6W6ABZc7cg?pwd=hwso 

extraction code 提取码：hwso

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

link 链接：https://pan.baidu.com/s/1qWiIM3jsU4vag6cdMVmORg?pwd=n0eb 
extraction code 提取码：n0eb 

该训练文件夹包含预训练模型和正式阶段模型的参数文件。

pretrain_pth文件夹为预训练模型的参数文件。其中，epoch_88_loss_4.852962.pth为多聚焦、多曝光、医学图像融合任务所使用的预训练模型参数文件，epoch_76_loss_0.666261.pth为红外和可见光图像融合任务所使用的预训练模型参数文件。

formal_model_pth文件夹为正式阶段模型的参数文件。其中，epoch_144_loss_0.514635.pth为多聚焦、医学图像融合任务所使用的正式阶段模型参数文件，epoch_46_loss_1.127614.pth为多曝光、红外和可见光图像融合任务所使用的正式阶段模型参数文件。

This training folder contains the parameter files for the pretrain model and the formal phase model.

The pretrain_pth folder is the parameter file for the pretrained model. Among them, epoch_88_loss_4.852962.pth is the parameter file of the pre-training model used for the multi-focus, multi-exposure, medical image fusion task, and epoch_76_loss_0.666261.pth is the parameter file of the pre-training model used for the infrared and visible light image fusion task.

The formal_model_pth folder is the parameter file of the formal phase model. Among them, epoch_144_loss_0.514635.pth is the parameter file of the formal stage model used for the multi-focus, medical image fusion task, and epoch_46_loss_1.127614.pth is the parameter file of the formal stage model used for the multi-exposure, infrared and visible light image fusion task.

# 关于引用 About the citation
文章已上线，欢迎引用。 

The article is now online. Feel free to cite it.

Article Link 文章链接: https://www.sciencedirect.com/science/article/pii/S1077314224002996

@article{WANG2024104218,

title = {UUD-Fusion: An unsupervised universal image fusion approach via generative diffusion model},

journal = {Computer Vision and Image Understanding},

volume = {249},

pages = {104218},

year = {2024},

issn = {1077-3142},

doi = {https://doi.org/10.1016/j.cviu.2024.104218},

url = {https://www.sciencedirect.com/science/article/pii/S1077314224002996},

author = {Xiangxiang Wang and Lixing Fang and Junli Zhao and Zhenkuan Pan and Hui Li and Yi Li}
}
