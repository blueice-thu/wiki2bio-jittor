import jittor as jt
import numpy as np
from AttentionUnit import AttentionWrapper
from dualAttentionUnit import dualAttentionWrapper
from LstmUnit import LstmUnit
from fgateLstmUnit import fgateLstmUnit
from OutputUnit import OutputUnit

jt.flags.use_cuda = 1

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

        self.embedding = jt.nn.Embedding(self.source_vocab, self.emb_size)
        self.embedding.require_grad = True
        if self.field_concat or self.fgate_enc or self.encoder_add_pos or self.decoder_add_pos:
            self.fembedding = jt.nn.Embedding(self.field_vocab, self.field_size)
            self.fembedding.require_grad = True
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.pembedding = jt.nn.Embedding(self.position_vocab, self.pos_size)
            self.pembedding.require_grad = True
            self.rembedding = jt.nn.Embedding(self.position_vocab, self.pos_size)
            self.rembedding.require_grad = True
        
        if self.field_concat or self.fgate_enc:
            self.params['fembedding'] = self.fembedding
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            self.params['pembedding'] = self.pembedding
            self.params['rembedding'] = self.rembedding
        self.params['embedding'] = self.embedding


        if self.dual_att:
            print('dual attention mechanism used')
            self.att_layer = dualAttentionWrapper(self.hidden_size, self.hidden_size, self.field_attention_size)
        else:
            self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size)
            print('normal attention used')
        self.units['attention'] = self.att_layer

        self.optimizer = jt.nn.Adam(self.parameters(), learning_rate)
    
    def execute(self, x):
        """ 
        {self.encoder_input: x['enc_in'], self.encoder_len: x['enc_len'], 
        self.encoder_field: x['enc_fd'], self.encoder_pos: x['enc_pos'], 
        self.encoder_rpos: x['enc_rpos'], self.decoder_input: x['dec_in'],
        self.decoder_len: x['dec_len'], self.decoder_output: x['dec_out']}
        """
        self.encoder_input = x['enc_in'] = jt.array(x['enc_in'])
        encoder_embed = self.embedding(self.encoder_input)
        if x['dec_in'] != None:
            x['dec_in'] = jt.array(x['dec_in'])
            decoder_embed = self.embedding(x['dec_in'])
        if self.field_concat or self.fgate_enc or\
            self.encoder_add_pos or self.decoder_add_pos:
            x['enc_fd'] = jt.array(x['enc_fd'])
            field_embed = self.fembedding(x['enc_fd'])
            field_pos_embed = field_embed
            if self.field_concat:
                encoder_embed = jt.concat([encoder_embed, field_embed], dim=2)
        if self.position_concat or self.encoder_add_pos or self.decoder_add_pos:
            x['enc_pos'] = jt.array(x['enc_pos'])
            x['enc_rpos'] = jt.array(x['enc_rpos'])
            pos_embed = self.pembedding(x['enc_pos'])
            rpos_embed = self.rembedding(x['enc_rpos'])
            if self.position_concat:
                encoder_embed = jt.concat([encoder_embed, pos_embed, rpos_embed], dim=2)
                field_pos_embed = jt.concat([field_embed, pos_embed, rpos_embed], dim=2)
            elif self.encoder_add_pos or self.decoder_add_pos:
                field_pos_embed = jt.concat([field_embed, pos_embed, rpos_embed], dim=2)
        # ===== encode ===== #
        x['enc_len'] = jt.array(x['enc_len'])
        if self.fgate_enc:
            en_outputs, en_state = self.fgate_encoder(encoder_embed, field_pos_embed, x['enc_len'])
        else:
            en_outputs, en_state = self.encoder(self.encoder_embed, x['enc_len'])
        self.en_outputs = en_outputs
        self.field_pos_embed = field_pos_embed
        # ===== decode ===== #
        if x['dec_len'] != None:
            x['dec_len'] = jt.array(x['dec_len'])
            de_outputs, de_state = self.decoder_t(en_state, decoder_embed, x['dec_len'])
        self.g_tokens, self.atts = self.decoder_g(en_state)
        self.mean_loss = None
        if x['dec_out'] != None:
            x['dec_out'] = jt.array(x['dec_out'])
            batch_size, maxlen, _ = de_outputs.shape
            losses = jt.nn.cross_entropy_loss(
                output=de_outputs.reshape([batch_size*maxlen, -1]), 
                target=x['dec_out'].reshape([-1]),
                reduction='none'
            )
            losses = losses.reshape([batch_size, -1])
            mask = jt.nn.sign(jt.float32(x['dec_out']))
            losses = mask * losses
            self.mean_loss = jt.mean(losses)
        return self.mean_loss
        
    def encoder(self, inputs, inputs_len):
        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        hidden_size = self.hidden_size
        time = 0
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
        outputs = jt.transpose(jt.stack(emit_ta, dim=0), [1,0,2])
        return outputs, h0
    
    def fgate_encoder(self, inputs, fields, inputs_len):
        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        hidden_size = self.hidden_size

        time = 0
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
            finished = (t+1) >= inputs_len
            x_nt = jt.zeros([batch_size, self.uni_size], dtype=jt.float32) \
                if jt.all(finished) else inputs_ta[t+1]
            d_nt = jt.zeros([batch_size, self.field_attention_size], dtype=jt.float32) \
                if jt.all(finished) else fields_ta[t+1]
            return t+1, x_nt, d_nt, s_nt, finished
        
        while not jt.all(f0):
            time, inputs_ta[0], fields_ta[0], h0, f0 = loop_fn(time, inputs_ta[0], fields_ta[0], h0, emit_ta, f0)
        
        outputs = jt.transpose(jt.stack(emit_ta, dim=0), [1,0,2])
        return outputs, h0
    
    def decoder_t(self, initial_state, inputs, inputs_len):
        batch_size = inputs.shape[0]
        max_time = inputs.shape[1]
        time = 0
        h0 = initial_state
        f0 = jt.zeros([batch_size], dtype=jt.bool)
        x0 = self.embedding(jt.array([self.start_token] * batch_size))
        inputs_ta = jt.transpose(inputs, [1,0,2])
        emit_ta = []
        
        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
            o_t, _ = self.att_layer(o_t, self.en_outputs, self.field_pos_embed)
            o_t = self.dec_out(o_t, finished)
            emit_ta.append(o_t)
            finished = t >= inputs_len
            x_nt = jt.zeros([batch_size, self.emb_size], dtype=jt.float32) \
                if jt.all(finished) else inputs_ta[t]
            return t+1, x_nt, s_nt, finished

        while not jt.all(f0):
            time, x0, h0, f0 = loop_fn(time, x0, h0, emit_ta, f0)
        
        outputs = jt.transpose(jt.stack(emit_ta, dim=0), [1,0,2])
        return outputs, h0 
    
    def decoder_g(self, initial_state):
        batch_size = self.encoder_input.shape[0]
        encoder_len = self.encoder_input.shape[1]
        time = 0
        h0 = initial_state
        f0 = jt.zeros([batch_size], dtype=jt.bool)
        x0 = self.embedding(jt.array([self.start_token] * batch_size))
        emit_ta = []
        att_ta = []
        
        def loop_fn(t, x_t, s_t, emit_ta, att_ta, finished):
            o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
            o_t, w_t = self.att_layer(o_t, self.en_outputs, self.field_pos_embed)
            o_t = self.dec_out(o_t, finished)
            emit_ta.append(o_t)
            att_ta.append(w_t)
            next_token, _ = jt.argmax(o_t, dim=1)
            x_nt = self.embedding(next_token)
            finished = finished | (next_token == self.stop_token) \
                | (t >= self.max_length)
            return t+1, x_nt, s_nt, finished
        
        while not jt.all(f0):
            time, x0, h0, f0 = loop_fn(time, x0, h0, emit_ta, att_ta, f0)
        
        outputs = jt.transpose(jt.stack(emit_ta, dim=0), [1,0,2])
        pred_tokens, _ = jt.argmax(outputs, dim=2)
        atts = jt.stack(att_ta, dim=0)
        return pred_tokens, atts
    
    def decoder_beam(self, initial_state, beam_size):
        
        def beam_init():
            time_1 = 1
            beam_seqs_0 = jt.array([[self.start_token]] * beam_size)
            beam_probs_0 = jt.array([0.0] * beam_size)

            cand_seqs_0 = jt.array([[self.start_token]])
            cand_probs_0 = jt.array([-3e38])
            
            inputs = jt.array([self.start_token])
            x_t = self.embedding(inputs)
            print(x_t.shape)
            o_t, s_nt = self.dec_lstm(x_t, initial_state)
            o_t, w_t = self.att_layer(o_t, self.en_outputs, self.field_pos_embed)
            o_t = self.dec_out(o_t)
            print(s_nt[0].shape)
            logprobs2d = jt.nn.log_softmax(o_t)
            total_probs = logprobs2d + jt.reshape(beam_probs_0, [-1, 1])
            total_probs_noEOS = jt.concat([
                total_probs[0:1, 0:self.stop_token],
                jt.array([[-3e38]]),
                total_probs[0:1, self.stop_token+1:self.target_vocab]
            ], dim=1)
            flat_total_probs = jt.reshape(total_probs_noEOS, [-1])
            print(flat_total_probs.shape)
            
            beam_k = min(len(flat_total_probs), beam_size)
            top_indices, next_beam_probs = jt.argsort(flat_total_probs)
            top_indices = top_indices[:beam_k]
            next_beam_probs = next_beam_probs[:beam_k]
            
            next_bases = top_indices // self.target_vocab
            next_mods = jt.mod(top_indices, self.target_vocab)
            
            next_beam_seqs = jt.concat([
                beam_seqs_0[next_bases],
                jt.reshape(next_mods, [-1, 1])
            ], dim=1)
            
            cand_seqs_pad = jt.nn.pad(cand_seqs_0, [0,1])
            beam_seqs_EOS = jt.nn.pad(beam_seqs_0, [0,1])
            new_cand_seqs = jt.concat([cand_seqs_pad, beam_seqs_EOS], 0)
            print(new_cand_seqs.shape)
            
            EOS_probs = total_probs[0:beam_size, self.stop_token:self.stop_token+1]
            new_cand_probs = jt.concat([
                cand_probs_0,
                jt.reshape(EOS_probs, [-1])
            ], 0)
            cand_k = min(len(new_cand_probs), self.beam_size)
            next_cand_indices, next_cand_probs = jt.argsort(new_cand_probs)
            next_cand_indices = next_cand_indices[:cand_k]
            next_cand_probs = next_cand_probs[:cand_k]
            next_cand_seqs = new_cand_seqs[next_cand_indices]
            
            part_state_0 = jt.reshape(jt.stack([s_nt[0]] * beam_size), [beam_size, self.hidden_size])
            part_state_1 = jt.reshape(jt.stack([s_nt[1]] * beam_size), [beam_size, self.hidden_size])
            next_states = (part_state_0, part_state_1)
            print(part_state_0.shape)
            
            return next_beam_seqs, next_beam_probs, next_cand_seqs, \
                next_cand_probs, next_states, time_1
        
        beam_seqs_1, beam_probs_1, cand_seqs_1, \
            cand_probs_1, states_1, time_1 = beam_init()
        
        def beam_step(beam_seqs, beam_probs, cand_seqs, cand_probs, states, time):
            inputs = jt.reshape(beam_seqs[0:beam_size, time:time+1], [beam_size])
            x_t = self.embedding(inputs)
            o_t, s_nt = self.dec_lstm(x_t, states)
            o_t, w_t = self.att_layer(o_t, self.en_outputs, self.field_pos_embed)
            o_t = self.dec_out(o_t)
            
            logprobs2d = jt.nn.log_softmax(o_t)
            print(logprobs2d.shape)
            total_probs = logprobs2d + jt.reshape(beam_probs, [-1, 1])
            print(total_probs.shape)
            total_probs_noEOS = jt.concat([
                total_probs[0:beam_size, 0:self.stop_token],
                jt.reshape(jt.array([-3e38] * beam_size), [beam_size, 1]),
                total_probs[0:beam_size, self.stop_token+1:self.target_vocab]
            ], dim=1)
            flat_total_probs = jt.reshape(total_probs_noEOS, [-1])
            print(flat_total_probs.shape)
            
            beam_k = min(len(flat_total_probs), beam_size)
            top_indices, next_beam_probs = jt.argsort(flat_total_probs)
            top_indices = top_indices[:beam_k]
            next_beam_probs = next_beam_probs[:beam_k]
            
            next_bases = top_indices // self.target_vocab
            next_mods = jt.mod(top_indices, self.target_vocab)
            
            next_beam_seqs = jt.concat([
                beam_seqs[next_bases],
                jt.reshape(next_mods, [-1, 1])
            ], dim=1)
            
            next_states = (s_nt[0][next_bases], s_nt[1][next_bases])
            
            cand_seqs_pad = jt.nn.pad(cand_seqs, [0,1])
            beam_seqs_EOS = jt.nn.pad(beam_seqs, [0,1])
            new_cand_seqs = jt.concat([cand_seqs_pad, beam_seqs_EOS], 0)
            print(new_cand_seqs.shape)
            
            EOS_probs = total_probs[0:beam_size, self.stop_token:self.stop_token+1]
            new_cand_probs = jt.concat([
                cand_probs,
                jt.reshape(EOS_probs, [-1])
            ], 0)
            cand_k = min(len(new_cand_probs), self.beam_size)
            next_cand_indices, next_cand_probs = jt.argsort(new_cand_probs)
            next_cand_indices = next_cand_indices[:cand_k]
            next_cand_probs = next_cand_probs[:cand_k]
            next_cand_seqs = new_cand_seqs[next_cand_indices]
            
            return next_beam_seqs, next_beam_probs, next_cand_seqs, \
                next_cand_probs, next_states, time+1
        
        def beam_cond(beam_probs, beam_seqs, cand_probs, cand_seqs, state, time):
            length = (beam_probs.max() >= cand_probs.min())
            return length and (time < 60)
        
        loop_vars = [beam_seqs_1, beam_probs_1, cand_seqs_1, cand_probs_1, states_1, time_1]
        
        while beam_cond(*loop_vars):
            loop_vars = beam_step(*loop_vars)
        
        return loop_vars[0], loop_vars[1], loop_vars[2], loop_vars[3]
            
    def generate(self, x):
        self.execute(x)
        return self.g_tokens, self.atts

    def save(self, path):
        for u in self.units:
            self.units[u].save(path + u + ".pkl")
        jt.save(self.params, path + self.name + ".pkl")
    
    def load(self, path):
        for u in self.units:
            self.units[u].load(path + u + ".pkl")
        params = jt.load(path + self.name + ".pkl")
        for param in params:
            self.params[param].assign(params[param])


def embedding_lookup(params, ids):
    # TODO: implement tf.embedding_lookup
    # REF: https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup
    return jt.array()
