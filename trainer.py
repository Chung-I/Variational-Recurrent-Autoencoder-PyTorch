from __future__ import division

import onmt
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import opts
import kenlm


def plot_stats(save_model):

    def _plot_stats(stats):
        metrics = ['train_loss', 'train_KLD', 'train_KLD_obj', 'train_accuracy',
        'valid_loss', 'valid_KLD', 'valid_accuracy', 'valid_lm_nll']
        model_dir = os.path.dirname(save_model)

        for metric in metrics:
            plt.plot(stats['step'], stats[metric])
            plt.xlabel("step")
            if "accuracy" in metric:
                plt.ylabel("percentage")
            else:
                plt.ylabel("nats/word")
            plt.title(metric.replace("_", " "))
            plt.savefig(os.path.join(model_dir, metric + ".jpg"))
            plt.close('all')
        plt.plot(stats['kl_rate'])
        plt.xlabel("step")
        plt.ylabel("percentage")
        plt.title("KL rate")
        plt.savefig(os.path.join(model_dir,"kl_rate.jpg"))
        plt.close('all')

    return _plot_stats

def NMTCriterion(vocabSize, gpus):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if gpus:
        crit.cuda()
    return crit

def KLDLoss(kl_min):
    def kld_loss(mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar).mul_(-0.5)
        kl_min_vec = Variable(KLD_element.data.new(KLD_element.size()).fill_(kl_min))
        KLD = torch.sum(KLD_element)
        KLD_obj_element = torch.max(KLD_element, kl_min_vec)
        KLD_obj = torch.sum(KLD_obj_element)
        return KLD, KLD_obj

    return kld_loss
def memoryEfficientLoss(max_generator_batches):
    def _memoryEfficientLoss(outputs, targets, crit, eval=False):
        # compute generations one piece at a time
        num_correct, loss = 0, 0

        batch_size = outputs.size(1)
        outputs_split = torch.split(outputs, max_generator_batches)
        targets_split = torch.split(targets, max_generator_batches)
        for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
            out_t = out_t.view(-1, out_t.size(2))
            loss_t = crit(out_t, targ_t.view(-1))
            pred_t = out_t.max(1)[1]
            num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(onmt.Constants.PAD).data).sum()
            num_correct += num_correct_t
            loss += loss_t

        grad_output = None if outputs.grad is None else outputs.grad.data
        return loss, grad_output, num_correct

    return _memoryEfficientLoss

def plotTsne(epoch, save_model):
    def _plot_tsne(mus):
        mus_embedded = TSNE().fit_transform(mus.data.cpu().numpy())
        plt.gca().set_aspect('equal', adjustable='box')
        plt.scatter(mus_embedded[:, 0], mus_embedded[:, 1])

        model_dir = os.path.dirname(save_model)
        fig_name = "tsne_mu_epoch_{}.jpg".format(epoch)
        plt.savefig(os.path.join(model_dir, fig_name))

    return _plot_tsne

def eval(model, criterion, plot_tsne, tsne_num_batches):
    def _eval(data):
        total_loss = 0
        total_KLD = 0
        total_words = 0
        total_num_correct = 0

        model.eval()

        mus = []
        for i in range(len(data)):
            batch = data[i][:-1] # exclude original indices
            outputs, mu, logvar = model(batch)
            if i < tsne_num_batches:
                mus.append(mu)
            targets = batch[1][1:]  # exclude <s> from targets
            _memoryEfficientLoss = memoryEfficientLoss(opt.max_generator_batches)
            loss, _, num_correct = _memoryEfficientLoss(
                    outputs, targets, criterion, eval=True)

            KLD, KLD_obj = KLDLoss(0)(mu, logvar)

            total_loss += loss.data[0]
            total_KLD += KLD.data[0]
            total_num_correct += num_correct
            total_words += targets.data.ne(onmt.Constants.PAD).sum()

        mus = torch.cat(mus, 0)
        plot_tsne(mus)

        return total_loss / total_words, total_KLD / total_words, total_num_correct / total_words

    return _eval

def get_nll(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=False, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    nll = total_nll/total_wc
    return nll

def trainModel(model, trainData, validData, dataset, optim, stats, opt):
    print(model)

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt'].size(), opt.gpus)
    translator = onmt.Translator(opt)
    lm = kenlm.Model(opt.lm_path)

    start_time = time.time()

    def trainEpoch(epoch):

        model.train()

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_KLD, total_KLD_obj, total_words, total_num_correct = 0, 0, 0, 0, 0
        report_loss, report_KLD, report_KLD_obj, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0, 0, 0
        start = time.time()
        for i in range(len(trainData)):

            total_step = epoch * len(trainData) + i
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1] # exclude original indices

            model.zero_grad()
            outputs, mu, logvar = model(batch, total_step)
            targets = batch[1][1:]  # exclude <s> from targets
            _memoryEfficientLoss = memoryEfficientLoss(opt.max_generator_batches)
            loss, gradOutput, num_correct = _memoryEfficientLoss(
                    outputs, targets, criterion)

            KLD, KLD_obj = KLDLoss(opt.kl_min)(mu, logvar)
            if opt.k != 0:
                kl_rate = 1 / (1 + opt.k * math.exp(-total_step/opt.k))
            else:
                kl_rate = 1
            KLD_obj = kl_rate * KLD_obj

            elbo = KLD_obj + loss
            elbo.backward()

            # update the parameters
            optim.step()

            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss.data[0]
            report_KLD += KLD.data[0]
            report_KLD_obj += KLD_obj.data[0]
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += sum(batch[0][1])
            total_loss += loss.data[0]
            total_KLD += KLD.data[0]
            total_KLD_obj += KLD_obj.data[0]
            total_num_correct += num_correct
            total_words += num_words
            stats['kl_rate'].append(kl_rate)
            if i % opt.log_interval == -1 % opt.log_interval:
                print("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; KLD: %6.2f; KLD obj: %6.2f; kl rate: %2.6f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                      (epoch, i+1, len(trainData),
                      report_num_correct / report_tgt_words * 100,
                      math.exp(report_loss / report_tgt_words),
                      report_KLD / report_tgt_words,
                      report_KLD_obj / report_tgt_words,
                      kl_rate,
                      report_src_words/(time.time()-start),
                      report_tgt_words/(time.time()-start),
                      time.time()-start_time))
                mu_mean = mu.mean()
                mu_std  = mu.std()
                logvar_mean = logvar.mean()
                logvar_std = logvar.std()
                print("mu mean: {:0.5f}".format(mu_mean.data[0]))
                print("mu std: {:0.5f}".format(mu_std.data[0]))
                print("logvar mean: {:0.5f}".format(logvar_mean.data[0]))
                print("logvar std: {:0.5f}".format(logvar_std.data[0]))
                report_loss = report_KLD = report_KLD_obj = report_tgt_words = report_src_words = report_num_correct = 0

                start = time.time()

        return total_loss / total_words, total_KLD / total_words, total_KLD_obj / total_words, total_num_correct / total_words

    best_valid_acc = max(stats['valid_accuracy']) if stats['valid_accuracy'] else 0
    best_valid_ppl = math.exp(min(stats['valid_loss'])) if stats['valid_loss'] else math.inf
    best_valid_lm_nll = math.exp(min(stats['valid_lm_nll'])) if stats['valid_lm_nll'] else math.inf
    best_epoch = 1 + np.argmax(stats['valid_accuracy']) if stats['valid_accuracy'] else 1
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_loss, train_KLD, train_KLD_obj, train_acc = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))

        stats['train_loss'].append(train_loss)
        stats['train_KLD'].append(train_KLD)
        stats['train_KLD_obj'].append(train_KLD_obj)
        stats['train_accuracy'].append(train_acc)

        print('Train perplexity: %g' % train_ppl)
        print('Train KL Divergence: %g' % train_KLD)
        print('Train KL divergence objective: %g' % train_KLD_obj)
        print('Train accuracy: %g' % (train_acc*100))

        #  (2) evaluate on the validation set

        plot_tsne = plotTsne(epoch, opt.save_model)
        _eval = eval(model, criterion, plot_tsne, opt.tsne_num_batches)
        valid_loss, valid_KLD, valid_acc = _eval(validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        sampled_sentences = []
        for i in range(opt.validation_num_batches):
            predBatch, predScore = translator.sample(opt.batch_size)
            for pred in predBatch:
            sampled_sentences.append(" ".join(pred[0]))
        valid_lm_nll = get_nll(lm, sampled_sentences)



        stats['valid_loss'].append(valid_loss)
        stats['valid_KLD'].append(valid_KLD)
        stats['valid_accuracy'].append(valid_acc)
        stats['valid_lm_nll'].append(valid_lm_nll)
        stats['step'].append(epoch * len(trainData))

        print('Validation perplexity: %g' % valid_ppl)
        print('Validation KL Divergence: %g' % valid_KLD)
        print('Validation accuracy: %g' % (valid_acc*100))
        print('Validation kenlm nll: %g' % (valid_lm_nll))

        #  (3) plot statistics
        _plot_stats = plot_stats(opt.save_model)
        _plot_stats(stats)

        #  (4) update the learning rate
        optim.updateLearningRate(valid_loss, epoch)
        if best_valid_lm_nll > valid_lm_nll: # only store checkpoints if accuracy improved
            if epoch > opt.start_epoch:
                os.remove('%s_acc_%.2f_ppl_%.2f_lmnll_%.2f_e%d.pt'\
                % (opt.save_model, 100*best_valid_acc, best_valid_ppl, best_valid_lm_nll, best_epoch))
            best_valid_acc = valid_acc
            best_valid_lm_nll = valid_lm_nll
            best_valid_ppl = valid_ppl
            best_epoch = epoch
            model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
            #  (5) drop a checkpoint
            checkpoint = {
                'model': model_state_dict,
                'dicts': dataset['dicts'],
                'opt': opt,
                'epoch': epoch,
                'optim': optim,
                'stats': stats
            }
            torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_lmnll_%.2f_e%d.pt' % (opt.save_model, 100*valid_acc, valid_ppl, valid_lm_nll, epoch))

    return best_valid_lm_nll


def train(opt, dataset):

    if torch.cuda.is_available() and not opt.gpus:
        print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

    if opt.gpus:
        cuda.set_device(opt.gpus[0])
        opt.cuda = True
    else:
        opt.cuda = False

    ckpt_path = opt.train_from
    if ckpt_path:
        print('Loading dicts from checkpoint at %s' % ckpt_path)
        checkpoint = torch.load(ckpt_path)
        opt = checkpoint['opt']

    print("Loading data from '%s'" % opt.data)

    if ckpt_path:
        dataset['dicts'] = checkpoint['dicts']
    model_dir = os.path.dirname(opt.save_model)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')
    assert dicts['src'].size() == dicts['tgt'].size()
    dict_size = dicts['src'].size()
    word_lut = nn.Embedding(dicts['src'].size(),
                            opt.word_vec_size,
                            padding_idx=onmt.Constants.PAD)
    generator = nn.Sequential(
        nn.Linear(opt.rnn_size, dicts['tgt'].size()),
        nn.LogSoftmax())
    encoder = onmt.Models.Encoder(opt, word_lut)
    decoder = onmt.Models.Decoder(opt, word_lut, generator)

    model = onmt.Models.NMTModel(encoder, decoder, opt)


    if ckpt_path:
        print('Loading model from checkpoint at %s' % ckpt_path)
        model.load_state_dict(checkpoint['model'])
        opt.start_epoch = checkpoint['epoch'] + 1

    if len(opt.gpus) >= 1:
        model.cuda()
    else:
        model.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

    if not ckpt_path:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        encoder.load_pretrained_vectors(opt)
        decoder.load_pretrained_vectors(opt)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
        optim.set_parameters(model.parameters())
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        optim.set_parameters(model.parameters())
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())


    if ckpt_path:
        stats = checkpoint['stats']
    else:
        stats = {'train_loss': [], 'train_KLD': [], 'train_KLD_obj': [],
        'train_accuracy': [], 'kl_rate': [], 'valid_loss': [], 'valid_KLD': [],
        'valid_accuracy': [], 'valid_lm_nll', 'step': []}

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    best_valid_lm_nll = trainModel(model, trainData, validData, dataset, optim, stats, opt)
    return best_valid_lm_nll


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train.py')

    opts.model_opts(parser)
    opts.train_opts(parser)
    opt = parser.parse_args()
    train(opt)
