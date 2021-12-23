# This Python file uses the following encoding: utf-8
import argparse
import os
import sys

import numpy as np
from PyQt5 import QtWidgets, QtCore
from form import Ui_Widget
from preProcess import Vocab
from SeqUnit import SeqUnit

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size',    default=500,    type=int, help='Size of each layer.')
parser.add_argument('--emb_size',       default=400,    type=int, help='Size of embedding.')
parser.add_argument('--field_size',     default=50,     type=int, help='Size of embedding.')
parser.add_argument('--pos_size',       default=5,      type=int, help='Size of embedding.')
parser.add_argument('--batch_size',     default=32,     type=int, help='Batch size of train set.')
parser.add_argument('--epoch',          default=50,     type=int, help='Number of training epoch.')
parser.add_argument('--source_vocab',   default=20003,  type=int, help='vocabulary size')
parser.add_argument('--field_vocab',    default=1480,   type=int, help='vocabulary size')
parser.add_argument('--position_vocab', default=31,     type=int, help='vocabulary size')
parser.add_argument('--target_vocab',   default=20003,  type=int, help='vocabulary size')
parser.add_argument('--report',         default=5000,   type=int, help='report valid results after some steps')
parser.add_argument('--learning_rate',  default=0.0003, type=float, help='learning rate')

parser.add_argument('--mode',   default='train',            type=str, help='train or test')
parser.add_argument('--load',   default='0',                type=str, help='load directory')  # BBBBBESTOFAll
parser.add_argument('--dir',    default='processed_data',   type=str, help='data set directory')
parser.add_argument('--limits', default=0,                  type=int, help='max data set size')

parser.add_argument('--dual_attention', default=True, type=bool, help='dual attention layer or normal attention')
parser.add_argument('--fgate_encoder',  default=True, type=bool, help='add field gate in encoder lstm')

parser.add_argument('--field',          default=False,  type=bool, help='concat field information to word embedding')
parser.add_argument('--position',       default=False,  type=bool, help='concat position information to word embedding')
parser.add_argument('--encoder_pos',    default=True,   type=bool, help='position information in field-gated encoder')
parser.add_argument('--decoder_pos',    default=True,   type=bool, help='position information in dual attention decoder')

args = parser.parse_args()

class MyWindow(QtWidgets.QMainWindow):

    def __init__(self, model):
        super().__init__()
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.header = []
        self.content = []
        self.summary = ""
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
        self.summary = self.ui.summary.toPlainText()
        self.generate()

        self.header = []
        self.content = []
        self.summary = ""
        self.ui.input_display.clear()
        self.ui.summary.clear()

    def generate(self):
        self.model.eval()

        x = {'enc_in': [], 'enc_fd': [], 'enc_pos': [], 'enc_rpos': [], 'enc_len': [],
             'dec_in': [], 'dec_len': [], 'dec_out': []}

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
        summary = [self.v.word2id(t) for t in list(self.summary.strip().split())]
        gold = summary + [2]

        summary_len = len(summary)
        text_len = len(text)

        x['enc_in'].append(text)
        x['enc_len'].append(text_len)
        x['enc_fd'].append(field)
        x['enc_pos'].append(pos)
        x['enc_rpos'].append(rpos)
        x['dec_in'].append(summary)
        x['dec_len'].append(summary_len)
        x['dec_out'].append(gold)

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
    model = SeqUnit(batch_size=args.batch_size, hidden_size=args.hidden_size, emb_size=args.emb_size,
                    field_size=args.field_size, pos_size=args.pos_size, field_vocab=args.field_vocab,
                    source_vocab=args.source_vocab, position_vocab=args.position_vocab,
                    target_vocab=args.target_vocab, name="seq2seq",
                    field_concat=args.field, position_concat=args.position,
                    fgate_enc=args.fgate_encoder, dual_att=args.dual_attention, decoder_add_pos=args.decoder_pos,
                    encoder_add_pos=args.encoder_pos, learning_rate=args.learning_rate)
    save_dir = 'results/res/' + args.load + '/'
    model.load(save_dir)

    app = QtWidgets.QApplication([])
    MainWindow = MyWindow(model)
    MainWindow.setWindowTitle("wiki2bio demo")
    MainWindow.show()

    sys.exit(app.exec_())
