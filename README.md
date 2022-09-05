# ICDM 2022: 大规模电商图上的风险商品检测 代码文档

## Requirement

操作系统 Ubuntu 20.04.4 LTS (GNU/Linux 5.4.0-117-generic x86_64), Python 3.7.13, Pytorch 1.10.1, pyg 2.0.4, 具体的所有环境依赖见 ``code/requirement.txt``。

需安装损失函数包：
```
pip install info-nce-pytorch
```

GPU 依赖：CUDA 11.3, CUDNN 8.4


## Reimplement Result

### Session 1

```
cd code
```

Generate Pyg Dataset
```
sh format_pyg_session1.sh
```

Inference
```
sh session1_inference.sh
```

Training
```
sh session1_train.sh
```


### Session 2
```
cd code
```

Generate Pyg Dataset
```
sh format_pyg_session2.sh
```

Inference
```
sh session2_inference.sh
```

Training
```
sh session2_train.sh
```

