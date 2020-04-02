- tensorflow==1.14.0
- keras==2.2.4
- numpy==1.16.2
- pandas==0.23.4
- python==3.6

任务介绍：

    文本二分类任务。对文档进行分类，判断是否属于政治上的出访类事件。


### 数据集介绍：
    训练集：280个样本，测试集：60个样本

### 模型：

    ALBERT作为特征提取，模型采用最简单的神经网络模型：DNN. 直接运行model_train.py即可。

### 模型效果：

1. 在训练集的acc为0.9857,在测试集上的acc为0.9500

2. model_predict.py中的预测语句预测全部正确。

### 训练模型存储文件：
visit_classify.h5

### HTTP服务
server.py

