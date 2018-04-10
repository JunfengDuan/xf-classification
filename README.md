# 基于CNN实现的分类模型

## Train
### 信访目的模型训练

python3 train_run.py --ckpt_path=purpose_ckpt --corpus_path=xf_purpose_data

### 扬言模型训练

python3 train_run.py --ckpt_path=yy_ckpt --corpus_path=xf_yy_data


## Todo：
1.f1-score

2.model 调用接口

3.以预训练（pre-train）的word2vec向量初始化词向量，训练过程中调整词向量，能加速收敛

4.RCNN

5.数据重整

6.类目不均衡问题:尝试类似 booststrap 方法调整 loss 中样本权重方式解决。

7.输出准确率-误差图

cat xf_test.txt| head -n 300 >> ../sample.txt
