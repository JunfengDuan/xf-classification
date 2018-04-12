# 基于CNN实现的分类模型

## Train
### 参数修改
分类数：configuration.py num_classes

### 信访目的模型训练

python3 train_run.py --ckpt_path=purpose_ckpt --corpus_path=xf_purpose_data --model_name=purpose

### 扬言模型训练

python3 train_run.py --ckpt_path=yy_ckpt --corpus_path=xf_yy_data --model_name=yy

## Tensorboard 可视化

tensorboard --logdir=tensorboard/yy

## 模型测试
打开mian方法第二行注释，注释第一行

python3 yy_predict_run.py --model_test=true

## 模型预测,启动服务
python3 yy_predict_run.py

## 调用接口
访问：http://host:8084/yy_predict

传参格式{"text" : "..."}

返回数据格式：json字符串{"result":"是/否"}

## Todo：
1.f1-score

2.model 调用接口

3.以预训练（pre-train）的word2vec向量初始化词向量，训练过程中调整词向量，能加速收敛

4.RCNN

5.数据重整

6.类目不均衡问题:尝试类似 booststrap 方法调整 loss 中样本权重方式解决。

7.输出准确率-误差图

head -n 300 xf_test.txt > sample.txt
