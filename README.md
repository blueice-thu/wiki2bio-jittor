# wiki2bio-jittor

- 运行环境：Linux

- 处理过的数据集下载：https://wss1.cn/f/731ufp5g3c0。下载解压文件夹processed_data到本目录下。

original_data 也需要放在目录下。

# Requirements.txt

```
jittor
nltk
PyQt5
```

安装依赖：

```bash
perl -v
cpan -v
cpan install XML::DOM
apt-get install libxml-parser-perl
apt-cache search libdb
apt-get install libdb5.3-dev  # 可能需要更换版本
```

下载 ROUGE 1.5.5：https://github.com/summanlp/evaluation/tree/master/ROUGE-RELEASE-1.5.5

设置环境变量：

```bash
vim ~/.profile
export ROUGE_EVAL_HOME="$ROUGE_EVAL_HOME:~/ROUGE-1.5.5/data"
# vim ~/.bashrc
# export PATH="~/ROUGE-1.5.5/:$PATH"
# source ~/.bashrc
```

重新打开 terminal，替换 bug 文件：

```bash
cd ROUGE-1.5.5/data/
rm WordNet-2.0.exc.db
chmod +x ./WordNet-2.0-Exceptions/buildExeptionDB.pl
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
```

进行测试：

```bash
cd ROUGE-1.5.5
chmod +x ROUGE-1.5.5.pl
perl runROUGE-test.pl
```

安装 pyrouge: (不要从 pypi 安装，有 bug！已经安装了，需要先 uninstall 再执行以下步骤)

```bash
git clone https://github.com/bheinzerling/pyrouge
cd pyrouge
python setup.py install
python -m pyrouge.test
pyrouge_set_rouge_path ~/ROUGE-1.5.5
```

修改 `PythonROUGE.py` 中的代码为 ROUGE 的路径：

```python
    ROUGE_path = '~/ROUGE-1.5.5/ROUGE-1.5.5.pl'
    data_path = '~/ROUGE-1.5.5/data'
```

# 图形界面

- 将训练后保存的模型文件`model.pkl`放在文件夹`./results/res/demo`下
- 运行`python gui.py`
- 在图形界面中的`column_header`和`content`处逐行输入表头和内容，点击`add`按钮逐行添加
- 添加完成后，点击`submit`提交输入，在`result`处可以查看生成结果