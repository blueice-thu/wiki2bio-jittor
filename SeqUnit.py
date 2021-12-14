import jittor as jt
import numpy as np
from AttentionUnit import AttentionWrapper
from dualAttentionUnit import dualAttentionWrapper
from LstmUnit import LstmUnit
from fgateLstmUnit import fgateLstmUnit
from OutputUnit import OutputUnit

class SeqUnit(jt.Module):
    def __init__(self, batch_size, hidden_size, emb_size, field_size, pos_size, source_vocab, field_vocab,
            position_vocab, target_vocab, field_concat, position_concat, fgate_enc, dual_att,
            encoder_add_pos, decoder_add_pos, learning_rate, name, start_token=2, stop_token=2, max_length=150):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.field_size = field_size
        self.pos_size = pos_size
        self.uni_size = emb_size if not field_concat else emb_size+field_size
        self.uni_size = self.uni_size if not position_concat else self.uni_size+2*pos_size
        self.field_encoder_size = field_size if not encoder_add_pos else field_size+2*pos_size
        self.field_attention_size = field_size if not decoder_add_pos else field_size+2*pos_size
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.field_vocab = field_vocab
        self.position_vocab = position_vocab
        self.grad_clip = 5.0
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.name = name
        self.field_concat = field_concat
        self.position_concat = position_concat
        self.fgate_enc = fgate_enc
        self.dual_att = dual_att
        self.encoder_add_pos = encoder_add_pos
        self.decoder_add_pos = decoder_add_pos

        self.encoder_input = jt.int32([])
        self.encoder_field = jt.int32([])
        self.encoder_pos = jt.int32([])
        self.encoder_rpos = jt.int32([])
        self.decoder_input = jt.int32([])
        self.encoder_len = jt.int32([])
        self.decoder_len = jt.int32([])
        self.decoder_output = jt.int32([])
        self.enc_mask = jt.nn.sign(jt.float32(self.encoder_pos))

        self.params = {}

        if self.fgate_enc:
            self.enc_lstm = fgateLstmUnit(self.hidden_size, self.uni_size, self.field_encoder_size)
        else:
            self.enc_lstm = LstmUnit(self.hidden_size, self.uni_size)
        self.dec_lstm = LstmUnit(self.hidden_size, self.emb_size)
        self.dec_out = OutputUnit(self.hidden_size, self.target_vocab)

        self.units = {'encoder_lstm': self.enc_lstm,'decoder_lstm': self.dec_lstm, 'decoder_output': self.dec_out}

        self.embedding = jt.rand([self.source_vocab, self.emb_size])
        self.encoder_embed = embedding_lookup(self.embedding, self.encoder_input)
        self.decoder_embed = embedding_lookup(self.embedding, self.decoder_input)
        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            self.fembedding = jt.rand([self.field_vocab, self.field_size])
            self.field_embed = embedding_lookup(self.fembedding, self.encoder_field)
            self.field_pos_embed = self.field_embed
            if self.field_concat:
                self.encoder_embed = jt.contrib.concat([self.encoder_embed, self.field_embed], 2)
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.pembedding = jt.rand([self.position_vocab, self.pos_size])
            self.rembedding = jt.rand([self.position_vocab, self.pos_size])
            self.pos_embed = embedding_lookup(self.pembedding, self.encoder_pos)
            self.rpos_embed = embedding_lookup(self.rembedding, self.encoder_rpos)
            if position_concat:
                self.encoder_embed = jt.contrib.concat([self.encoder_embed, self.pos_embed, self.rpos_embed], 2)
                self.field_pos_embed = jt.contrib.concat([self.field_embed, self.pos_embed, self.rpos_embed], 2)
            elif self.encoder_add_pos or self.decoder_add_pos:
                self.field_pos_embed = jt.contrib.concat([self.field_embed, self.pos_embed, self.rpos_embed], 2)
        
        if self.field_concat or self.fgate_enc:
            self.params['fembedding'] = self.fembedding
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.params['pembedding'] = self.pembedding
            self.params['rembedding'] = self.rembedding
        self.params['embedding'] = self.embedding

        # ======================================== encoder ======================================== #
        if self.fgate_enc:
            print('field gated encoder used')
            en_outputs, en_state = self.fgate_encoder(self.encoder_embed, self.field_pos_embed, self.encoder_len)
        else:
            print('normal encoder used')
            en_outputs, en_state = self.encoder(self.encoder_embed, self.encoder_len)
        
        # ======================================== decoder ======================================== #
        if self.dual_att:
            print('dual attention mechanism used')
            self.att_layer = dualAttentionWrapper(self.hidden_size, self.hidden_size, self.field_attention_size, en_outputs, self.field_pos_embed)
        else:
            print('normal attention used')
            self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size, en_outputs)
        self.units['attention'] = self.att_layer

        de_outputs, de_state = self.decoder_t(en_state, self.decoder_embed, self.decoder_len)
        self.g_tokens, self.atts = self.decoder_g(en_state)

        losses = jt.nn.cross_entropy_loss(output=de_outputs, target=self.decoder_output)
        mask = jt.sign(jt.float32(self.decoder_output))
        losses = mask * losses
        self.mean_loss = jt.mean(losses)

        # TODO
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.grad_clip)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    
    def execute(self, x):
        """ 
        {self.encoder_input: x['enc_in'], self.encoder_len: x['enc_len'], 
        self.encoder_field: x['enc_fd'], self.encoder_pos: x['enc_pos'], 
        self.encoder_rpos: x['enc_rpos'], self.decoder_input: x['dec_in'],
        self.decoder_len: x['dec_len'], self.decoder_output: x['dec_out']}
        """
        encoder_embed = self.embedding[x['enc_in']]
        decoder_embed = self.embedding[x['dec_in']]
        if self.field_concat or self.fgate_enc or\
            self.encoder_add_pos or self.decoder_add_pos:
            field_embed = self.fembedding[x['enc_fd']]
            if self.field_concat:
                encoder_embed = jt.concat([encoder_embed, field_embed], dim=2)
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            pos_embed = self.pembedding[x['enc_pos']]
            rpos_embed = self.rembedding[x['enc_rpos']]
            if self.position_concat:
                encoder_embed = jt.concat([encoder_embed, pos_embed, rpos_embed], dim=2)
                field_pos_embed = jt.concat([field_embed, pos_embed, rpos_embed], dim=2)
            elif self.encoder_add_pos or self.decoder_add_pos:
                field_pos_embed = jt.concat([field_embed, pos_embed, rpos_embed], dim=2)
        # ===== encode ===== #
        if self.fgate_enc:
            en_outputs, en_state = self.fgate_encoder(encoder_embed, field_pos_embed, x['enc_len'])
        else:
            en_outputs, en_state = self.encoder(self.encoder_embed, x['enc_len'])
        # ===== decode ===== #
        de_outputs, de_state = self.decoder_t(en_state, decoder_embed, x['dec_len'])
        self.g_tokens, self.atts = self.decoder_g(en_state)
        losses = jt.nn.cross_entropy_loss(output=de_outputs, target=x['dec_out'])
        mask = jt.sign(jt.float32(x['dec_out']))
        losses = mask * losses
        self.mean_loss = jt.mean(losses)
        return self.mean_loss
        
    def encoder(self, inputs, inputs_len):
        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        hidden_size = self.hidden_size
        time = jt.array(0)
        h0 = (
            jt.zeros([batch_size, hidden_size], dtype=jt.float32),
            jt.zeros([batch_size, hidden_size], dtype=jt.float32)
        )
        f0 = jt.zeros([batch_size], dtype=jt.bool)
        # inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        # emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        inputs_ta = jt.transpose(inputs, [1,0,2])
        emit_ta = []
        
        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.enc_lstm(x_t, s_t, finished)
            emit_ta.append(o_t)
            finished = (t+1) >= inputs_len
            x_nt = jt.zeros([batch_size, self.uni_size], dtype=jt.float32) \
                if jt.all(finished) else inputs_ta[t+1]
            return t+1, x_nt, s_nt, finished

        while not jt.all(f0):
            time, inputs_ta[0], h0, f0 = loop_fn(time, inputs_ta[0], h0, emit_ta, f0)
        outputs = jt.transpose(jt.concat(emit_ta, dim=0), [1,0,2])
        return outputs, h0
    
    def fgate_encoder(self, inputs, fields, inputs_len):
        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        hidden_size = self.hidden_size

        time = jt.array(0)
        h0 = (
            jt.zeros([batch_size, hidden_size], dtype=jt.float32),
            jt.zeros([batch_size, hidden_size], dtype=jt.float32)
        )
        f0 = jt.zeros([batch_size], dtype=jt.bool)
        inputs_ta = jt.transpose(inputs, [1,0,2])
        fields_ta = jt.transpose(fields, [1,0,2])
        emit_ta = []

        def loop_fn(t, x_t, d_t, s_t, emit_ta, finished):
            o_t, s_nt = self.enc_lstm(x_t, d_t, s_t, finished)
            emit_ta.append(o_t)
            finished = t+1 >= inputs_len
            x_nt = jt.zeros([batch_size, self.uni_size], dtype=jt.float32) \
                if jt.all(finished) else inputs_ta[t+1]
            d_nt = jt.zeros([batch_size, self.field_attention_size], dtype=jt.float32) \
                if jt.all(finished) else fields_ta[t+1]
            return t+1, x_nt, d_nt, s_nt, finished
        
        while not jt.all(f0):
            time, inputs_ta[0], fields_ta[0], h0, f0 = loop_fn(time, inputs_ta[0], fields_ta[0], h0, emit_ta, f0)
        
        outputs = jt.transpose(jt.concat(emit_ta, dim=0), [1,0,2])
        return outputs, h0
    
    def decoder_t(self, initial_state, inputs, inputs_len):
        pass
            
    def generate(self, x):
        pass

    def save(self, path):
        jt.save(self.params, path)
    
    def load(self, path):
        params = jt.load(path)
        for param in params:
            self.params[param].assign(params[param])


def embedding_lookup(params, ids):
    # TODO: implement tf.embedding_lookup
    # REF: https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
    return jt.array()
