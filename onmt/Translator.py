import onmt
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

def slerp(low, high, val):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    low = low.numpy()
    high = high.numpy()
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return torch.from_numpy(np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high)



class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model)

        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']

        encoder = onmt.Models.Encoder(model_opt, self.src_dict)
        decoder = onmt.Models.Decoder(model_opt, self.tgt_dict)
        model = onmt.Models.NMTModel(encoder, decoder, model_opt)

        generator = nn.Sequential(
            nn.Linear(model_opt.rnn_size, self.tgt_dict.size()),
            nn.LogSoftmax())

        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

        if opt.cuda:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator

        self.model = model
        self.model.eval()


    def buildData(self, srcBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD) for b in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                       onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD,
                       onmt.Constants.EOS_WORD) for b in goldBatch]

        return onmt.Dataset(srcData, tgtData,
            self.opt.batch_size, self.opt.cuda, volatile=True)

    def buildTargetTokens(self, pred, src):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        return tokens

    def beam_decode(self, encStates):
        batchSize = encStates.size(0)
        beamSize = self.opt.beam_size
        rnnSize = self.model.decoder.hidden_size
        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]
        decStates = self.model.prelu_dec(self.model.latent_to_decoder(encStates))
        decStates = decStates.view(self.model.layers, decStates.size(0), -1)
        decStates = torch.split(decStates, decStates.size(-1)//2, 2)

        decStates = (Variable(decStates[0].data.repeat(1, beamSize, 1)),
                     Variable(decStates[1].data.repeat(1, beamSize, 1)))
        context = Variable(encStates.data.repeat(beamSize, 1))
        decOut = self.model.make_init_decoder_output(context)


        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.opt.max_sent_length):
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                               if not b.done]).t().contiguous().view(1, -1)

            decOut, decStates = self.model.decoder(Variable(input, volatile=True), decStates, decOut)
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            out = self.model.generator.forward(decOut)

            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(
                        -1, beamSize, remainingSents, decState.size(2))[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx) \
                                    .view(*newSize), volatile=True)

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)

            remainingSents = len(active)

        #  (4) package everything up

        allHyp, allScores = [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            hyps = [beam[b].getHyp(k) for k in ks[:n_best]]
            allHyp += [hyps]
        return allHyp, allScores


    def translateBatch(self, srcBatch, tgtBatch):
        batchSize = srcBatch[0].size(1)
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        encStates, _ = self.model.encode(srcBatch)
        srcBatch = srcBatch[0] # drop the lengths needed for encoder

        rnnSize = self.model.decoder.hidden_size

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = encStates.data.new(batchSize).zero_()
        if tgtBatch is not None:
            decStates = encStates

            decOut = self.model.decode(
                decStates, tgtBatch[:-1])
            for dec_t, tgt_t in zip(decOut, tgtBatch[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores

        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.

        beam = [onmt.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]
        decStates = self.model.prelu_dec(self.model.latent_to_decoder(encStates))
        decStates = decStates.view(self.model.layers, decStates.size(0), -1)
        decStates = torch.chunk(decStates, 2, 2)
        decStates = (Variable(decStates[0].data.repeat(1, beamSize, 1)),
                     Variable(decStates[1].data.repeat(1, beamSize, 1)))
        context = Variable(encStates.data.repeat(beamSize, 1))
        decOut = self.model.make_init_decoder_output(context)

        padMask = srcBatch.data.eq(onmt.Constants.PAD).t().unsqueeze(0).repeat(beamSize, 1, 1)

        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.opt.max_sent_length):

            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                               if not b.done]).t().contiguous().view(1, -1)

            decOut, decStates = self.model.decoder(Variable(input, volatile=True), decStates, decOut)
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            out = self.model.generator.forward(decOut)

            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx]):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(
                        -1, beamSize, remainingSents, decState.size(2))[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx) \
                                    .view(*newSize), volatile=True)

            decStates = (updateActive(decStates[0]), updateActive(decStates[1]))
            decOut = updateActive(decOut)
            padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        #  (4) package everything up

        allHyp, allScores = [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            hyps = [beam[b].getHyp(k) for k in ks[:n_best]]
            allHyp += [hyps]

        #return allHyp, allScores, allAttn, goldScores
        return allHyp, allScores, goldScores

    def translate(self, srcBatch, goldBatch):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        src, tgt, indices = dataset[0]

        #  (2) translate
        pred, predScore, goldScore = self.translateBatch(src, tgt)
        pred, predScore, goldScore = list(zip(*sorted(zip(pred, predScore, goldScore, indices), key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b])
                        for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, goldScore

    def interpolate(self, srcBatch, num_pts):
        dataset = self.buildData(srcBatch, [])
        src, _, indices = dataset[0]
        mu, logvar = self.model.encode(src)
        start, end = mu[0].data, mu[1].data
        points = Variable(torch.cat([slerp(start, end, w).view(1, -1) for w in torch.range(0, 1, 1/(num_pts - 1))], 0))
        pred, predScore = self.beam_decode(points)
        indices = [i for i in range(0, num_pts)] if indices[0] < indices[1] else [i for i in range(num_pts-1, -1, -1)]
        pred, predScore = list(zip(*sorted(zip(pred, predScore, indices), key=lambda x: x[-1])))[:-1]
        predBatch = []
        for b in range(points.size(0)):
            predBatch.append(
                [self.tgt_dict.convertToLabels(pred[b][n], onmt.Constants.EOS)[:-1]
                    for n in range(self.opt.n_best)]
            )
        return predBatch, predScore

    def sample(self, num_pts):
        points = Variable(self.tt.FloatTensor(num_pts, self.model.latent_size).normal_())
        pred, predScore = self.beam_decode(points)
        indices = [i for i in range(0, num_pts)] 
        pred, predScore = list(zip(*sorted(zip(pred, predScore, indices), key=lambda x: x[-1])))[:-1]
        predBatch = []
        for b in range(points.size(0)):
            predBatch.append(
                [self.tgt_dict.convertToLabels(pred[b][n], onmt.Constants.EOS)[:-1]
                    for n in range(self.opt.n_best)]
            )
        return predBatch, predScore
        
