# wiki2bio-jittor

- 运行环境：Linux

- 处理过的数据集下载：https://wss1.cn/f/731ufp5g3c0。下载解压文件夹processed_data到本目录下。

- TODO：模型代码。

# Requirements.txt

```
jittor
nltk
```

# TODO

完成 SeqUnit.py 。

一些函数 jittor 和 numpy 均没有实现，需要手动实现：
- embedding_lookup

可能遇到的问题是，tensorflow 使用计算图 `sess.run()` 进行运算，需要对于一些代码的位置进行调整。
