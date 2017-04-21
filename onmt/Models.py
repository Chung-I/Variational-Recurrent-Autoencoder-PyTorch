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
        self.rnn = nn.LSTM(input_size, opt.rnn_size,
                num_layers=opt.layers,
                dropout=opt.dropout)
        
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden):
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)

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
        concat_hidden_size = opt.layers * opt.rnn_size * 2  # multiply by 2 because it's LSTM
        self.encoder_to_mu = nn.Linear(concat_hidden_size, opt.latent_size)
        self.encoder_to_logvar = nn.Linear(concat_hidden_size., opt.latent_size)
        self.latent_to_decoder = nn.Linear(self.config.latent_size, self.config.encoder_hidden_size)
        self.elu = nn.ELU()

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def encode(self, x):
        enc_hidden, _ = self.encoder(x)
        enc_hidden.transpose(0,1) # convert to batch major
        enc_hidden = enc_hidden.view(enc_hidden.size()[0], -1)
        mu = self.elu(self.encoder_to_mu(enc_hidden))
        logvar = self.elu(self.encoder_to_logvar(encoder_state))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul_(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z, tgt):
        decoder_state = self.elu(self.latent_to_decoder(z))
        # the decoder state is batch x (layers * rnn_size)
        # we need to convert it to layers x batch x (directions * dim)
        decoder_state.view(opt.layers, decoder_state.size()[0], -1)
        x, dec_hidden = self.decoder(tgt, decoder_state)
        return x

    def forward(self, input):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs
        mu, logvar = self.encode(src)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z, tgt)


        return out, mu, logvar
