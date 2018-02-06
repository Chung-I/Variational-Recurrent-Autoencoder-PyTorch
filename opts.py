def train_opts(parser):

    # Model loading/saving options

    parser.add_argument('-data', required=True,
                        help='Path to the *-train.pt file from preprocess.py')
    parser.add_argument('-save_model', default='model',
                        help="""Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the
                        validation perplexity""")


    # Init options
    parser.add_argument('-start_epoch', type=int, default=1,
                        help='The epoch from which to start')
    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution
                        with support (-param_init, param_init)""")
    parser.add_argument('-train_from', default='', type=str,
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""")
    
    # Optimization options
    parser.add_argument('-batch_size', type=int, default=64,
                        help='Maximum batch size')
    parser.add_argument('-max_generator_batches', type=int, default=32,
                        help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but uses
                        more memory.""")
    parser.add_argument('-epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument('-max_grad_norm', type=float, default=5,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.0,
                        help='Dropout probability; applied between LSTM stacks.')
    parser.add_argument('-curriculum', action="store_true",
                        help="""For this many epochs, order the minibatches based
                        on source sequence length. Sometimes setting this to 1 will
                        increase convergence speed.""")
    parser.add_argument('-extra_shuffle', action="store_true",
                        help="""By default only shuffle mini-batch order; when true,
                        shuffle and re-assign mini-batches""")
    parser.add_argument('-k', type=int, default=50000,
                        help='sigmoid increase rate for kl rate. r = 1 / (1 + k * exp(-i/k))')
    parser.add_argument('-ss', type=int, default=50000,
                        help='sigmoid increase rate for scheduled sampling.')
    parser.add_argument('-deterministic', action='store_true',
                        help='if true, no reparameterization')
    parser.add_argument('-kl_min', type=float, default=0.0,
                        help='Minmum kl divergence for each latent dimension')
    parser.add_argument('-feed_gt_prob', type=float, default=0.75,
                        help="""Probability of feeding ground truth when training. See word dropout in \"Generating Sentences from a continuous space\".""")
    parser.add_argument('-dynamic_decode', action='store_true',
                        help='feed outputs of previous steps instead of ground truth')
    
    #learning rate
    parser.add_argument('-learning_rate', type=float, default=1e-4,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
    parser.add_argument('-learning_rate_decay', type=float, default=1,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=8,
                        help="""Start decaying every epoch after and including this
                        epoch""")
    
    #pretrained word vectors
    parser.add_argument('-pre_word_vecs_enc',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side.
                        See README for specific formatting instructions.""")
    parser.add_argument('-pre_word_vecs_dec',
                        help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side.
                        See README for specific formatting instructions.""")
    
    # GPU
    parser.add_argument('-gpus', default=[], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    
    #log
    parser.add_argument('-log_interval', type=int, default=50,
                        help="Print stats at this interval.")
    parser.add_argument('-tsne_num_batches', type=int, default=5,
                        help="How many batches to be added into tsne visualization")

def model_opts(parser):
    
    # Model options

    parser.add_argument('-layers', type=int, default=2,
                        help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('-rnn_size', type=int, default=500,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int, default=500,
                        help='Word embedding sizes')
    parser.add_argument('-latent_size', type=int, default=16,
                        help='Latent space sizes')
    parser.add_argument('-input_feed', type=int, default=1,
                        help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder.""")
    parser.add_argument('-brnn', action='store_true',
                        help='Use a bidirectional encoder')
    parser.add_argument('-brnn_merge', default='concat',
                        help="""Merge action for the bidirectional hidden states:
                        [concat|sum]""")
    parser.add_argument('-prelu', action='store_true',
                        help='Use prelu between encoder and decoder')
    
