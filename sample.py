from __future__ import division

import onmt
import torch
import argparse
import math
import pdb

parser = argparse.ArgumentParser(description='interpolate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-num_pts', type=int, default=10,
                    help='number of output points of interpolation.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")



def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))

def addone(f):
    for line in f:
        yield line
    yield None

def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator(opt)

    outF = open(opt.output, 'w')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    count = 0

    predBatch, predScore = translator.sample(opt.num_pts)
 
    #predScoreTotal += sum(score[0] for score in predScore)
    predWordsTotal += sum(len(x[0]) for x in predBatch)
    #if tgtF is not None:
    #    goldScoreTotal += sum(goldScore)
    #    goldWordsTotal += sum(len(x) for x in tgtBatch)

    for b in range(len(predBatch)):
        count += 1
        outF.write(" ".join(predBatch[b][0]) + '\n')
        outF.flush()

        if opt.verbose:
            #srcSent = ' '.join(srcBatch[b])
            #if translator.tgt_dict.lower:
            #    srcSent = srcSent.lower()
            #print('SENT %d: %s' % (count, srcSent))
            print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
            print("PRED SCORE: %.4f" % predScore[b][0])

            if opt.n_best > 1:
                print('\nBEST HYP:')
                for n in range(opt.n_best):
                    print("[%.4f] %s" % (predScore[b][n], " ".join(predBatch[b][n])))

            print('')

if __name__ == "__main__":
    main()
