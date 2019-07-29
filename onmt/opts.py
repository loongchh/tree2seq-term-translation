""" Implementation of all available options """
from __future__ import print_function

import argparse
from onmt.models.sru import CheckSRU


def model_opts(parser):
    """
    These options are passed to the construction of the model.
    Be careful with these as they will be used during translation.
    """

    # Embedding Options
    group = parser.add_argument_group('Model-Embeddings')
    group.add_argument(
        '--src-word-vec-size',
        type=int,
        default=300,
        help='Word embedding size for src.')
    group.add_argument(
        '--tgt-word-vec-size',
        type=int,
        default=300,
        help='Word embedding size for tgt.')
    group.add_argument(
        '--word-vec-size',
        type=int,
        default=-1,
        help='Word embedding size for src and tgt.')

    group.add_argument(
        '--share-decoder-embeddings',
        action='store_true',
        help="""Use a shared weight matrix for the input and
                output word  embeddings in the decoder.""")
    group.add_argument(
        '--share-embeddings',
        action='store_true',
        help="""Share the word embeddings between encoder
                and decoder. Need to use shared dictionary for this
                option.""")
    group.add_argument(
        '--position-encoding',
        action='store_true',
        help="""Use a sin to mark relative words positions.
                Necessary for non-RNN style models.
                """)

    group = parser.add_argument_group('Model-Embedding Features')
    group.add_argument(
        '--feat-merge',
        type=str,
        default='concat',
        choices=['concat', 'sum', 'mlp'],
        help="""Merge action for incorporating features embeddings.
                Options [concat|sum|mlp].""")
    group.add_argument(
        '--feat-vec-size',
        type=int,
        default=-1,
        help="""If specified, feature embedding sizes
                will be set to this. Otherwise, feat-vec-exponent
                will be used.""")
    group.add_argument(
        '--feat-vec-exponent',
        type=float,
        default=0.7,
        help="""If -feat-merge-size is not set, feature
                embedding sizes will be set to N^feat-vec-exponent
                where N is the number of values the feature takes.""")

    # Encoder-decoder options
    group = parser.add_argument_group('Model- Encoder-Decoder')
    group.add_argument(
        '--model-type',
        default='text',
        help="""Type of source model to use. Allows
                the system to incorporate non-text inputs.
                Options are [text|img|audio].""")

    group.add_argument(
        '--encoder-type',
        type=str,
        default='rnn',
        choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn', 'tree'],
        help="""Type of encoder layer to use. Non-RNN layers
                are experimental. Options are [tree].""")
    group.add_argument(
        '--decoder-type',
        type=str,
        default='rnn',
        choices=['rnn', 'transformer', 'cnn', 'tree'],
        help="""Type of decoder layer to use. Non-RNN layers
                are experimental. Options are [seq|tree].""")

    group.add_argument(
        '--layers', type=int, default=-1, help='Number of layers in enc/dec.')
    group.add_argument(
        '--enc-layers',
        type=int,
        default=1,
        help='Number of layers in the encoder')
    group.add_argument(
        '--dec-layers',
        type=int,
        default=1,
        help='Number of layers in the decoder')
    group.add_argument(
        '--rnn-size', type=int, default=300, help='Size of rnn hidden states')
    group.add_argument(
        '--cnn-kernel-width',
        type=int,
        default=3,
        help="""Size of windows in the cnn, the kernel-size is
                       (cnn-kernel-width, 1) in conv layer""")

    group.add_argument(
        '--input-feed',
        type=int,
        default=1,
        help="""Feed the context vector at each time step as
                additional input (via concatenation with the word
                embeddings) to the decoder.""")
    group.add_argument(
        '--bridge',
        action="store_true",
        help="""Have an additional layer between the last encoder
                state and the first decoder state""")
    group.add_argument(
        '--rnn-type',
        type=str,
        default='LSTM',
        choices=['LSTM'],
        action=CheckSRU,
        help="""The gate type to use in the RNNs""")
    # group.add_argument('--residual',   action="store_true",
    #                     help="Add residual connections between RNN layers.")

    group.add_argument(
        '--context-gate',
        type=str,
        default=None,
        choices=['source', 'target', 'both'],
        help="""Type of context gate to use.
                Do not select for no context gate.""")

    # Attention options
    group = parser.add_argument_group('Model-Attention')
    group.add_argument(
        '--global-attention',
        type=str,
        default='general',
        choices=['dot', 'general', 'mlp'],
        help="""The attention type to use:
                dotprod or general (Luong) or MLP (Bahdanau)""")
    group.add_argument(
        '--global-attention-function',
        type=str,
        default="softmax",
        choices=["softmax", "sparsemax"])
    group.add_argument(
        '--self-attn-type',
        type=str,
        default="scaled-dot",
        help="""Self attention type in Transformer decoder
                layer -- currently "scaled-dot" or "average" """)
    group.add_argument(
        '--heads',
        type=int,
        default=12,
        help='Number of heads for transformer self-attention')
    group.add_argument(
        '--transformer-ff',
        type=int,
        default=2048,
        help='Size of hidden transformer feed-forward')

    # Generator and loss options.
    group.add_argument(
        '--copy-attn', action="store_true", help='Train copy attention layer.')
    group.add_argument(
        '--generator-function',
        default="log_softmax",
        choices=["log_softmax", "sparsemax"],
        help="""Which function to use for generating
                       probabilities over the target vocabulary (choices:
                       log_softmax, sparsemax)""")
    group.add_argument(
        '--copy-attn-force',
        action="store_true",
        help='When available, train to copy.')
    group.add_argument(
        '--reuse-copy-attn',
        action="store_true",
        help="Reuse standard attention for copy")
    group.add_argument(
        '--tree-combine',
        action="store_true",
        help="Combine linear and tree-LSTM hidden states with a tree.")
    group.add_argument(
        '--copy-loss-by-seqlength',
        action="store_true",
        help="Divide copy loss by length of sequence")
    group.add_argument(
        '--coverage-attn',
        action="store_true",
        help='Train a coverage attention layer.')
    group.add_argument(
        '--lambda-coverage',
        type=float,
        default=1,
        help='Lambda value for coverage.')


def preprocess_opts(parser):
    """ Pre-processing options """
    # Data options
    group = parser.add_argument_group('Data')
    group.add_argument(
        '--data-type',
        default="text",
        help="""Type of the source input.
                Options are [text|img].""")

    group.add_argument(
        '--train-src', required=True, help="Path to the training source data")
    group.add_argument(
        '--train-tgt', required=True, help="Path to the training target data")
    group.add_argument(
        '--valid-src',
        required=True,
        help="Path to the validation source data")
    group.add_argument(
        '--valid-tgt',
        required=True,
        help="Path to the validation target data")
    group.add_argument(
        '--test-src',
        required=True,
        help="Path to the testing source data")
    group.add_argument(
        '--test-tgt',
        required=True,
        help="Path to the testing target data")
    group.add_argument(
        '--train-src-parse', help="Path to the training source parse")
    group.add_argument(
        '--valid-src-parse', help="Path to the validation source parse")
    group.add_argument(
        '--test-src-parse', help="Path to the testing source parse")

    group.add_argument(
        '--src-dir',
        default="",
        help="Source directory for image or audio files.")

    group.add_argument(
        '--save-data', required=True, help="Output file for the prepared data")

    group.add_argument(
        '-max_shard_size',
        type=int,
        default=0,
        help="""Deprecated use shard_size instead""")
    group.add_argument(
        '--shard-size',
        type=int,
        default=0,
        help="""Divide src_corpus and tgt_corpus into
                       smaller multiple src_copus and tgt corpus files, then
                       build shards, each shard will have
                       opt.shard_size samples except last shard.
                       shard_size=0 means no segmentation
                       shard_size>0 means segment dataset into multiple shards,
                       each shard has shard_size samples""")

    # Dictionary options, for text corpus
    group = parser.add_argument_group('Vocab')
    group.add_argument(
        '--src-vocab',
        default="",
        help="""Path to an existing source vocabulary. Format:
                one word per line.""")
    group.add_argument(
        '--tgt-vocab',
        default="",
        help="""Path to an existing target vocabulary. Format:
                one word per line.""")
    group.add_argument(
        '--features-vocabs-prefix',
        type=str,
        default='',
        help="Path prefix to existing features vocabularies")
    group.add_argument(
        '--src-vocab-size',
        type=int,
        default=50000,
        help="Size of the source vocabulary")
    group.add_argument(
        '--tgt-vocab-size',
        type=int,
        default=50000,
        help="Size of the target vocabulary")

    group.add_argument(
        '--pre-word-vecs-src',
        help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the encoder side.
                       See README for specific formatting instructions.""")
    group.add_argument(
        '--pre-word-vecs-tgt',
        help="""If a valid path is specified, then this will load
                       pretrained word embeddings on the decoder side.
                       See README for specific formatting instructions.""")

    group.add_argument('--src-words-min-frequency', type=int, default=0)
    group.add_argument('--tgt-words-min-frequency', type=int, default=0)

    group.add_argument(
        '--dynamic-dict',
        action='store_true',
        help="Create dynamic dictionaries")
    group.add_argument(
        '--share-vocab',
        action='store_true',
        help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add_argument(
        '--src-seq-length',
        type=int,
        default=50,
        help="Maximum source sequence length")
    group.add_argument(
        '--src-seq-length-trunc',
        type=int,
        default=0,
        help="Truncate source sequence length.")
    group.add_argument(
        '--tgt-seq-length',
        type=int,
        default=50,
        help="Maximum target sequence length to keep.")
    group.add_argument(
        '--tgt-seq-length-trunc',
        type=int,
        default=0,
        help="Truncate target sequence length.")
    group.add_argument('--lower', action='store_true', help='lowercase data')

    # Data processing options
    group = parser.add_argument_group('Random')
    group.add_argument('--shuffle', type=int, default=1, help="Shuffle data")
    group.add_argument('--seed', type=int, default=3435, help="Random seed")

    group = parser.add_argument_group('Logging')
    group.add_argument(
        '--log-file',
        type=str,
        help="Output logs to a file under this path.")

    # Options most relevant to speech
    group = parser.add_argument_group('Speech')
    group.add_argument(
        '-sample_rate', type=int, default=16000, help="Sample rate.")
    group.add_argument(
        '-window_size',
        type=float,
        default=.02,
        help="Window size for spectrogram in seconds.")
    group.add_argument(
        '-window_stride',
        type=float,
        default=.01,
        help="Window stride for spectrogram in seconds.")
    group.add_argument(
        '-window',
        default='hamming',
        help="Window type for spectrogram generation.")

    # Option most relevant to image input
    group.add_argument(
        '-image_channel_size',
        type=int,
        default=3,
        choices=[3, 1],
        help="""Using grayscale image can training
                       model faster and smaller""")


def train_opts(parser):
    """ Training and saving options """
    group = parser.add_argument_group('General')
    group.add_argument(
        '--data',
        required=True,
        help="""Path prefix to the ".train.pt" and
                       ".valid.pt" file path from preprocess.py""")
    group.add_argument(
        '--checkpoint-dir',
        default='',
        help="""Model filename (the model will be saved as
                       <checkpoint-dir>-N.pt where N is the number
                       of steps""")
    group.add_argument(
        '--checkpoint-steps',
        type=int,
        default=5000,
        help="""Save a checkpoint every X steps""")
    group.add_argument(
        '--keep-checkpoint',
        type=int,
        default=-1,
        help="""Keep X checkpoints (negative: keep all)""")

    # GPU
    group.add_argument(
        '--gpuid',
        default=[],
        nargs='+',
        type=int,
        help="Deprecated see world_size and gpu_ranks.")
    group.add_argument(
        '--gpu-ranks',
        default=[],
        nargs='+',
        type=int,
        help="list of ranks of each process.")
    group.add_argument(
        '--world-size',
        default=1,
        type=int,
        help="total number of distributed processes.")
    group.add_argument(
        '--gpu-backend',
        default='nccl',
        nargs='+',
        type=str,
        help="Type of torch distributed backend")
    group.add_argument(
        '--gpu-verbose-level',
        default=0,
        type=int,
        help="Gives more info on each process per GPU.")
    group.add_argument(
        '--master-ip',
        default="localhost",
        type=str,
        help="IP of master for torch.distributed training.")
    group.add_argument(
        '--master-port',
        default=10000,
        type=int,
        help="Port of master for torch.distributed training.")
    group.add_argument(
        '--seed',
        type=int,
        default=-1,
        help="""Random seed used for the experiments
                       reproducibility.""")

    # Init options
    group = parser.add_argument_group('Initialization')
    group.add_argument(
        '--param-init',
        type=float,
        default=0.1,
        help="""Parameters are initialized over uniform distribution
                with support (-param-init, param-init).
                Use 0 to not use initialization""")
    group.add_argument(
        '--param-init-glorot',
        action='store_true',
        help="""Init parameters with xavier-uniform.
                Required for transfomer.""")

    group.add_argument(
        '--train-from',
        default='',
        type=str,
        help="""If training from a checkpoint then this is the
                path to the pretrained model's state-dict.""")

    # Pretrained word vectors
    # group.add_argument('--pre-word-vecs-src',
    #                    help="""If a valid path is specified, then this will load
    #                    pretrained word embeddings on the source side.
    #                    See README for specific formatting instructions.""")
    # group.add_argument('--pre-word-vecs-tgt',
    #                    help="""If a valid path is specified, then this will load
    #                    pretrained word embeddings on the target side.
    #                    See README for specific formatting instructions.""")
    # Fixed word vectors
    group.add_argument(
        '--fix-word-vecs-src',
        action='store_true',
        help="Fix word embeddings on the source.")
    group.add_argument(
        '--fix-word-vecs-tgt',
        action='store_true',
        help="Fix word embeddings on the target.")

    # Optimization options
    group = parser.add_argument_group('Optimization- Type')
    group.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Maximum batch size for training')
    group.add_argument(
        '--batch-type',
        default='sents',
        choices=["sents", "tokens"],
        help="""Batch grouping for batch-size. Standard
                is sents. Tokens will do dynamic batching""")
    group.add_argument(
        '--normalization',
        default='sents',
        choices=["sents", "tokens"],
        help='Normalization method of the gradient.')
    group.add_argument(
        '--accum-count',
        type=int,
        default=1,
        help="""Accumulate gradient this many times.
                Approximately equivalent to updating
                batch-size * accum-count batches at once.
                Recommended for Transformer.""")
    group.add_argument(
        '--valid-steps',
        type=int,
        default=1000,
        help='Perfom validation every X steps')
    group.add_argument(
        '--valid-batch-size',
        type=int,
        default=32,
        help='Maximum batch size for validation')
    group.add_argument(
        '--max-generator-batches',
        type=int,
        default=32,
        help="""Maximum batches of words in a sequence to run
                the generator on in parallel. Higher is faster, but
                uses more memory.""")
    group.add_argument(
        '--train-steps',
        type=int,
        default=20000,
        help='Number of training steps')
    group.add_argument(
        '--epochs', type=int, default=0, help='Deprecated, see --train-steps')
    group.add_argument(
        '--optim',
        default='adam',
        choices=['sgd', 'adagrad', 'adadelta', 'adam', 'sparseadam'],
        help="""Optimization method.""")
    group.add_argument(
        '--adagrad-accumulator-init',
        type=float,
        default=0,
        help="""Initializes the accumulator values in adagrad.
                Mirrors the initial-accumulator-value option
                in the tensorflow adagrad (use 0.1 for their default).
                """)
    group.add_argument(
        '--max-grad-norm',
        type=float,
        default=5,
        help="""If the norm of the gradient vector exceeds this,
                renormalize it to have the norm equal to
                max-grad-norm""")
    group.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help="Dropout probability; applied in LSTM stacks.")
    group.add_argument(
        '--truncated-decoder', type=int, default=0, help="""Truncated bptt.""")
    group.add_argument(
        '--adam-beta1',
        type=float,
        default=0.9,
        help="""The beta1 parameter used by Adam.
                Almost without exception a value of 0.9 is used in
                the literature, seemingly giving good results,
                so we would discourage changing this value from
                the default without due consideration.""")
    group.add_argument(
        '--adam-beta2',
        type=float,
        default=0.999,
        help="""The beta2 parameter used by Adam.
                Typically a value of 0.999 is recommended, as this is
                the value suggested by the original paper describing
                Adam, and is also the value adopted in other frameworks
                such as Tensorflow and Kerras, i.e. see:
                https://www.tensorflow.org/api-docs/python/tf/train/AdamOptimizer
                https://keras.io/optimizers/ .
                Whereas recently the paper "Attention is All You Need"
                suggested a value of 0.98 for beta2, this parameter may
                not work well for normal models / default
                baselines.""")
    group.add_argument(
        '--label-smoothing',
        type=float,
        default=0.0,
        help="""Label smoothing value epsilon.
                Probabilities of all non-true labels
                will be smoothed by epsilon / (vocab-size - 1).
                Set to zero to turn off label smoothing.
                For more detailed information, see:
                https://arxiv.org/abs/1512.00567""")
    # learning rate
    group = parser.add_argument_group('Optimization- Rate')
    group.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help="""Starting learning rate.
                Recommended settings: sgd = 1, adagrad = 0.1,
                adadelta = 1, adam = 0.001""")
    group.add_argument(
        '--learning-rate-decay',
        type=float,
        default=0.5,
        help="""If update-learning-rate, decay learning rate by
                this much if (i) perplexity does not decrease on the
                validation set or (ii) steps have gone past
                start-decay-steps""")
    group.add_argument(
        '--start-decay-steps',
        type=int,
        default=10000,
        help="""Start decaying every decay-steps after
                start-decay-steps""")
    group.add_argument(
        '--decay-steps',
        type=int,
        default=1000,
        help="""Decay every decay-steps""")

    group.add_argument(
        '--decay-method',
        type=str,
        default="",
        choices=['noam'],
        help="Use a custom decay rate.")
    group.add_argument(
        '--warmup-steps',
        type=int,
        default=4000,
        help="""Number of warmup steps for custom decay.""")

    group = parser.add_argument_group('Logging')
    group.add_argument(
        '--report-every',
        type=int,
        default=50,
        help="Print stats at this interval.")
    group.add_argument(
        '--log-file',
        type=str,
        help="Output logs to a file under this path.")
    group.add_argument(
        '--exp-host',
        type=str,
        default="",
        help="Send logs to this crayon server.")
    group.add_argument(
        '--exp',
        type=str,
        default="",
        help="Name of the experiment for logging.")
    group.add_argument(
        '--json',
        action='store_true',
        help="""Output training metrics in JSON format.""")

    # Use TensorboardX for visualization during training
    group.add_argument(
        "--tensorboard-dir",
        type=str,
        help="""Log directory for Tensorboard.
                This is also the name of the run.
                """)

    group = parser.add_argument_group('Speech')
    # Options most relevant to speech
    group.add_argument(
        '--sample-rate', type=int, default=16000, help="Sample rate.")
    group.add_argument(
        '--window-size',
        type=float,
        default=.02,
        help="Window size for spectrogram in seconds.")

    # Option most relevant to image input
    group.add_argument(
        '--image-channel-size',
        type=int,
        default=3,
        choices=[3, 1],
        help="""Using grayscale image can training
                       model faster and smaller""")


def translate_opts(parser):
    """ Translation / inference options """
    group = parser.add_argument_group('Model')
    group.add_argument(
        '--model',
        dest='models',
        metavar='MODEL',
        nargs='+',
        type=str,
        default=[],
        required=True,
        help='Path to model .pt file(s). '
        'Multiple models can be specified, '
        'for ensemble decoding.')
    group = parser.add_argument_group('Data')
    group.add_argument(
        '--data-type',
        default="text",
        help="Type of the source input. Options: [text|img].")

    group.add_argument(
        '--src',
        required=True,
        help="""Source sequence to decode (one line per
                sequence)""")
    group.add_argument(
        '--src-dir',
        default="",
        help='Source directory for image or audio files')
    group.add_argument(
        '--src-parse', help="Path to the source parse")
    group.add_argument('--tgt', help='True target sequence (optional)')
    group.add_argument(
        '--output',
        default='pred.txt',
        help="""Path to output the predictions (each line will
                be the decoded sequence""")
    group.add_argument(
        '--report-bleu',
        action='store_true',
        help="""Report bleu score after translation,
                call tools/multi_bleu.perl on command line""")
    group.add_argument(
        '--report-rouge',
        action='store_true',
        help="""Report rouge 1/2/3/L/SU4 score after translation
                       call tools/test_rouge.py on command line""")
    group.add_argument(
        '--report-f1',
        action='store_true',
        help="""Report F1/precision/recall scores after translation
                       call tools/test_rouge.py on command line""")

    # Options most relevant to summarization.
    group.add_argument(
        '--dynamic-dict',
        action='store_true',
        help="Create dynamic dictionaries")
    group.add_argument(
        '--share-vocab',
        action='store_true',
        help="Share source and target vocabulary")

    group = parser.add_argument_group('Beam')
    group.add_argument(
        '--fast',
        action="store_true",
        help="""Use fast beam search (some features may not be
                supported!)""")
    group.add_argument('--beam-size', type=int, default=5, help='Beam size')
    group.add_argument(
        '--min-length', type=int, default=0, help='Minimum prediction length')
    group.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum prediction length.')
    group.add_argument(
        '--max-sent-length',
        action=DeprecateAction,
        help="Deprecated, use `--max-length` instead")
    # Alpha and Beta values for Google Length + Coverage penalty
    # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
    group.add_argument(
        '--stepwise-penalty',
        action='store_true',
        help="""Apply penalty at every decoding step.
                Helpful for summary penalty.""")
    group.add_argument(
        '--length-penalty',
        default='none',
        choices=['none', 'wu', 'avg'],
        help="""Length Penalty to use.""")
    group.add_argument(
        '--coverage-penalty',
        default='none',
        choices=['none', 'wu', 'summary'],
        help="""Coverage Penalty to use.""")
    group.add_argument(
        '--alpha',
        type=float,
        default=0.,
        help="""Google NMT length penalty parameter
                (higher = longer generation)""")
    group.add_argument(
        '--beta',
        type=float,
        default=-0.,
        help="""Coverage penalty parameter""")
    group.add_argument(
        '--block-ngram-repeat',
        type=int,
        default=0,
        help='Block repetition of ngrams during decoding.')
    group.add_argument(
        '--ignore-when-blocking',
        nargs='+',
        type=str,
        default=[],
        help="""Ignore these strings when blocking repeats.
                You want to block sentence delimiters.""")
    group.add_argument(
        '--replace-unk',
        action="store_true",
        help="""Replace the generated UNK tokens with the
                source token that had highest attention weight. If
                phrase-table is provided, it will lookup the
                identified source token and give the corresponding
                target token. If it is not provided(or the identified
                source token does not exist in the table) then it
                will copy the source token""")

    group = parser.add_argument_group('Logging')
    group.add_argument(
        '--verbose',
        action="store_true",
        help='Print scores and predictions for each sentence')
    group.add_argument(
        '--json',
        action='store_true',
        help="""Output training metrics in JSON format.""")
    group.add_argument(
        '--log-file',
        type=str,
        help="Output logs to a file under this path.")
    group.add_argument(
        '--attn-debug',
        action="store_true",
        help='Print best attn for each word')
    group.add_argument(
        '--dump-beam',
        type=str,
        default="",
        help='File to dump beam information to.')
    group.add_argument(
        '--n-best',
        type=int,
        default=1,
        help="""If verbose is set, will output the n-best
                decoded sentences""")

    group = parser.add_argument_group('Efficiency')
    group.add_argument('--batch-size', type=int, default=30, help='Batch size')
    group.add_argument('--gpu', type=int, default=-1, help="Device to run on")

    # Options most relevant to speech.
    group = parser.add_argument_group('Speech')
    group.add_argument(
        '--sample-rate', type=int, default=16000, help="Sample rate.")
    group.add_argument(
        '--window-size',
        type=float,
        default=.02,
        help='Window size for spectrogram in seconds')
    group.add_argument(
        '--window-stride',
        type=float,
        default=.01,
        help='Window stride for spectrogram in seconds')
    group.add_argument(
        '--window',
        default='hamming',
        help='Window type for spectrogram generation')

    # Option most relevant to image input
    group.add_argument(
        '--image-channel-size',
        type=int,
        default=3,
        choices=[3, 1],
        help="""Using grayscale image can training
                       model faster and smaller""")


def add_md_help_argument(parser):
    """ md help parser """
    parser.add_argument(
        '--md',
        action=MarkdownHelpAction,
        help='print Markdown-formatted help text and exit.')


# MARKDOWN boilerplate


# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        return ""

    def format_help(self):
        print(self._prog)
        self._root_section.heading = '# Options: %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self) \
            .start_section('### **%s**' % heading)

    def _format_action(self, action):
        if action.dest == "help" or action.dest == "md":
            return ""
        lines = []
        lines.append(
            '* **-%s %s** ' %
            (action.dest, "[%s]" % action.default if action.default else "[]"))
        if action.help:
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    """ MD help action """

    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


class DeprecateAction(argparse.Action):
    """ Deprecate action """

    def __init__(self, option_strings, dest, help=None, **kwargs):
        super(DeprecateAction, self).__init__(
            option_strings, dest, nargs=0, help=help, **kwargs)

    def __call__(self, parser, namespace, values, flag_name):
        help = self.help if self.mdhelp is not None else ""
        msg = "Flag '%s' is deprecated. %s" % (flag_name, help)
        raise argparse.ArgumentTypeError(msg)
