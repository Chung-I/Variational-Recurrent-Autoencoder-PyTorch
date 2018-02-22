from __future__ import print_function
from __future__ import division

import argparse
from bayes_opt import BayesianOptimization

import trainer
import opts

def train_with_hparams(opt, dataset):
    def _train(dropout, kl_min, rnn_size, word_vec_size, latent_size):
        opt.dropout = dropout
        opt.kl_min = kl_min
        opt.rnn_size = rnn_size
        opt.word_vec_size = word_vec_size
        opt.latent_size = latent_size
        return trainer.train(opt)

    return _train

if __name__ == "__main__":
    gp_params = {"alpha": 1e-5}
    parser = argparse.ArgumentParser(description='bayes_train.py')
    opts.model_opts(parser)
    opts.train_opts(parser)
    opts.translator_opts(parser)
    opts.bo_opts(parser)
    opt = parser.parse_args()

    dataset = torch.load(opt.data)

    hparams_range = {
        'dropout': (0.0, 0.5),
        'kl_min': (0.0, 8.0),
        'rnn_size': (128, 1024),
        'word_vec_size': (128, 512),
        'latent_size': (8, 256)
    }
    BO = BayesianOptimization(train_with_hparams(opt, dataset), hparams_range)

    BO.maximize(n_iter=10, **gp_params)
    print('-' * 53)

    print('Final Results')
    print('SVC: %f' % svcBO.res['max']['max_val'])
    print('RFC: %f' % rfcBO.res['max']['max_val'])
    with open(opt.bo_output_path, 'wb') as out_f:
        pickle.dump(BO.res, out_f, -1)
