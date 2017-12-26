import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            emb = pack(self.word_lut(input[0]), input[1])
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
        return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)  
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, init_output):
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat((emb_t, output), 1)

            output, hidden = self.rnn(emb_t, hidden)
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden


class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, opt):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.layers = opt.layers
        self.gpus = opt.gpus
        self.latent_size = opt.latent_size
        self.feed_gt_prob = opt.feed_gt_prob
        concat_hidden_size = opt.layers * opt.rnn_size * 2  # multiply by 2 because it's LSTM
        self.encoder_to_mu = nn.Linear(concat_hidden_size, opt.latent_size)
        self.encoder_to_logvar = nn.Linear(concat_hidden_size, opt.latent_size)
        self.latent_to_decoder = nn.Linear(opt.latent_size, concat_hidden_size)
        self.prelu_mu = nn.PReLU()
        self.prelu_logvar = nn.PReLU()
        self.prelu_dec = nn.PReLU()

    def encode(self, x):
        enc_hidden, _ = self.encoder(x)
        #  the encoder hidden is a tensor tuple with dimension (layers*directions) x batch x dim
        #  we need to convert it to batch x (2*layers*directions*dim)
        enc_hidden = torch.cat((enc_hidden[0], enc_hidden[1]), 2).transpose(0,1).contiguous() 
        enc_hidden = enc_hidden.view(enc_hidden.size(0), -1)
        mu = self.prelu_mu(self.encoder_to_mu(enc_hidden))
        logvar = self.prelu_logvar(self.encoder_to_logvar(enc_hidden))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(torch.mul(logvar, 0.5))
        if self.gpus:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return torch.add(torch.mul(eps, std), mu)
    
    def make_init_decoder_output(self, z):
        batch_size = z.size(0)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(z.data.new(*h_size).zero_(), requires_grad=False)

    def decode(self, z, tgt):
        decoder_state = torch.chunk(self.prelu_dec(self.latent_to_decoder(z)).view(self.layers, z.size(0), -1), 2, 2)
        # the decoder state is batch x (2*layers*directions*dim)
        # we need to convert it to a tensor tuple with dimesion layers x batch x (directions * dim)
        #decoder_state = torch.split(decoder_state, decoder_state.size(-1)//2, 2)

        init_output = self.make_init_decoder_output(z)

        x, dec_hidden = self.decoder(tgt, decoder_state, init_output)
        return x
    
    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def replace_by_unk(self, seqs, prob=0.75):
        seq_len = seqs.size(0)
        batch_size = seqs.size(1)
        if self.gpus:
            probs = Variable(torch.cuda.FloatTensor(seq_len - 1, batch_size).fill_(prob))
            ones = Variable(torch.cuda.FloatTensor(1, batch_size).fill_(1.0))
        else:
            probs = Variable(torch.FloatTensor(seq_len - 2, batch_size).fill_(prob))
            ones = Variable(torch.FloatTensor(1, batch_size).fill_(1.0))
        probs = torch.cat((ones, probs), 0)
        samples = torch.bernoulli(probs)
        added = (1 - samples) * onmt.Constants.UNK
        return torch.mul(seqs, samples.long()) + added.long()

    def forward(self, input):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        if self.training and self.feed_gt_prob < 1:
            tgt = self.replace_by_unk(tgt, self.feed_gt_prob)
        mu, logvar = self.encode(src)
        z = self.reparameterize(mu, logvar)
        out = self.decode(mu, tgt)
        #enc_hidden, _ = self.encoder(src)
        #enc_hidden, _ = self.encoder(x)
        #init_output = self.make_init_decoder_output(enc_hidden[0])
        #enc_hidden = torch.cat((enc_hidden[0], enc_hidden[1]), 2)
        #enc_hidden = enc_hidden.transpose(0,1).contiguous() # convert to batch major
        #enc_hidden = enc_hidden.view(enc_hidden.size(0), -1)
        #mu = self.prelu_mu(self.encoder_to_mu(enc_hidden))
        #decoder_state = self.prelu_dec(self.latent_to_decoder(mu))
        #decoder_state = decoder_state.view(self.layers, decoder_state.size(0), -1)
        #decoder_state = torch.chunk(decoder_state, 2, 2)

        #decoder_state = (self._fix_enc_hidden(decoder_state[0]),
        #                 self._fix_enc_hidden(decoder_state[1]))

        #out, dec_hidden = self.decoder(tgt, decoder_state, init_output)
        #out, dec_hidden = self.decoder(tgt, enc_hidden, init_output)

        return out, mu, logvar
