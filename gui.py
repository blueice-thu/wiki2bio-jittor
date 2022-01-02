# This Python file uses the following encoding: utf-8
import os
import sys

from PyQt5 import QtWidgets, QtCore
from form import Ui_Widget
from preProcess import Vocab
import jittor as jt

class MyWindow(QtWidgets.QMainWindow):

    def __init__(self, model):
        super().__init__()
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.header = []
        self.content = []
        self.model = model

        self.v = Vocab()
        self.ui.add.clicked.connect(self.add)
        self.ui.submit.clicked.connect(self.submit)

    def add(self):
        header = self.ui.column_header.text()
        content = self.ui.content.toPlainText()
        self.header.append(header)
        self.content.append(content)
        self.ui.column_header.clear()
        self.ui.content.clear()
        text = header + ": " + content
        self.ui.input_display.append(text)

    def submit(self):
        self.generate()

        self.header = []
        self.content = []
        self.ui.input_display.clear()

    def generate(self):
        self.model.eval()

        x = {'enc_in': [], 'enc_fd': [], 'enc_pos': [], 'enc_rpos': [], 'enc_len': [],
             'dec_in': None, 'dec_len': None, 'dec_out': None}

        tmp_text = [list(t.strip().split()) for t in self.content]
        tmp_text = [[self.v.word2id(w) for w in t] for t in tmp_text]
        tmp_field = [[self.v.key2id(self.header[i]) for _ in range(len(tmp_text[i]))] for i in range(len(tmp_text))]
        tmp_pos = [[i for i in range(1, len(t)+1)] for t in tmp_text]
        tmp_rpos = [[i for i in range(len(t), 0, -1)] for t in tmp_text]
        text, field, pos, rpos = [], [], [], []
        for i in range(len(tmp_text)):
            text.extend(tmp_text[i])
            field.extend(tmp_field[i])
            pos.extend(tmp_pos[i])
            rpos.extend(tmp_rpos[i])

        text_len = len(text)

        x['enc_in'].append(text)
        x['enc_len'].append(text_len)
        x['enc_fd'].append(field)
        x['enc_pos'].append(pos)
        x['enc_rpos'].append(rpos)

        predictions, _ = model.generate(x)
        summary = list(predictions.data[0])
        if 2 in summary:
            summary = summary[:summary.index(
                2)] if summary[0] != 2 else [2]
        unk_sum = []
        for tk, tid in enumerate(summary):
            unk_sum.append(self.v.id2word(tid))
        pred_unk = [str(x) for x in unk_sum]
        self.ui.result.setText(" ".join(pred_unk))


if __name__ == "__main__":
    save_dir = 'results/res/demo/model.pkl'
    model = jt.load(save_dir)

    app = QtWidgets.QApplication([])
    MainWindow = MyWindow(model)
    MainWindow.setWindowTitle("wiki2bio demo")
    MainWindow.show()

    sys.exit(app.exec_())
