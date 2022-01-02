# wiki2bio-jittor

- 运行环境：Linux

- 原数据集下载：https://cloud.tsinghua.edu.cn/f/c3e086f593ad42c2bce8/
  
  下载后解压文件夹`original_data`到本目录

- 处理过的数据集下载：https://cloud.tsinghua.edu.cn/f/1e8bd5fab0c24b5b87dc/
  
  下载后解压所有文件到`processed_data`文件夹下
  
- 训练好的模型下载：https://cloud.tsinghua.edu.cn/f/35ae21d27a2c493ab2c9/
  
  下载后解压`model.pkl`到`results/res/demo`文件夹下

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

# 训练和测试

训练：

```bash
python main.py
```

训练时会在路径 `/results/res/{timestamp}/loads/` 路径下报错模型，将选定的模型移动到 `/results/res/{timestamp}/` 路径下，进行测试：

```bash
# For BLEU
python main.py --mode test --load 1640709464365
# For ROUGE
python PythonROUGE.py
```

# 图形界面

- 将训练后保存的模型文件`model.pkl`放在文件夹`results/res/demo`下
- 运行`python gui.py`
- 在图形界面中的`column_header`和`content`处逐行输入表头和内容，点击`add`按钮逐行添加
- 添加完成后，点击`submit`提交输入，在`result`处可以查看生成结果