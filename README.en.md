\# CONTRAST\_LEARING-master



\#### Description

用于AD分类的双分支可学习模型。融合了局部-全局融合模型和有监督的对比学习，添加了小波变换的数学思想。



\#### Model Architecture

最终采用的模型是V10，位置在：CONTRAST\_LEARING-master\\Others\\structure\_image



\#### Datasets



1\.  输入一：全脑数据：CONTRAST\_LEARING-master\\datasets\\\\NIFTI5

2\.  输入二：海马数据：CONTRAST\_LEARING-master\\datasets\\hippdata



\#### Models



1\.  最终可以运行的高准确率的两个版本：ModelV23.py、ModelV24.py

2\.  modelV23.py——>双输入+对比学习的版本

3\.  modelV24.py——>小波变换+双输入+对比学习的版本



\#### 运行脚本main.py



1\.  在切换不同模型时，只需要更换导入包中的modelV24



\#### 最佳模型结果



1\. runs——>experiment\_20250719\_1604\_服务器\_0.9904\_小波方案一





