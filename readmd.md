# Introduction
项目博客：https://blog.csdn.net/qq_21120275/article/details/102159314  
源码地址：http://www.langpy.cn/resources/lstmcrfgxcq.zip  
数据来源于http://www.lishixinzhi.com 和 http://www.uuqgs.com/ , 博客作者自己手动标注数据在train.txt文件中，因此我们读取的是已经被BIO格式标注的数据

## 配置要求
作者给出的配置要求如下：
```base
tensorflow=1.14.0
keras=2.2.4
keras-contrib=2.0.8
keras-contrib安装参考：https://github.com/keras-team/keras-contrib
```
这里使用conda新建python版本为3.7的环境，并安装上述框架，同时安装`ipykernel`后使用命令`python -m ipykernel install --user --name py3.7 --display-name "py3.7"` 将conda环境写入jupyter的kernel中。

原有的train.txt文件中，文字已被标注，格式如下：
```base
太 B-SUBJECT
宗 I-SUBJECT
喜 B-PREDICATE
爱 I-PREDICATE
的 O
魏 O
王 O
李 B-OBJECT
泰 I-OBJECT
。 O

后 O
来 O
， O
李 O
承 O
乾 O
甚 O
至 O
```

## 问题及解决
在jupyter notebook中执行下面代码的时候：
```python
from keras_contrib.layers.crf import CRF, crf_loss, crf_viterbi_accuracy
model = load_model('kg-mode.model', custom_objects={"CRF": CRF, 'crf_loss': crf_loss,'crf_viterbi_accuracy': crf_viterbi_accuracy})
```
发现会报错keras包中的语句：“str object has no attribute decode”  
查询：https://zhuanlan.zhihu.com/p/368593126 得知可能是h5py版本过高的问题，查看自己3.7python环境h5py版本为3.7.0.
因此，卸载并重装2.10版本的h5py，若找不到安装源可在命令后加`-i https://pypi.tuna.tsinghua.edu.cn/simple/`，然后注意jupyter notebook中要执行重新引入相关包的那一步，问题解决！