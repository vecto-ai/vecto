from .language_modeling import Language_modeling as Benchmark


def add_extra_args(parser):
    parser.add_argument("embeddings")
#    parser.add_argument("dataset", default="ptb")
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--test", default=True,
                        help='use small test dataset')
    parser.add_argument("--method",
                        default='lstm',
                        choices=['lr', '2FFNN', 'lstm'],
                        help='name of method')
    parser.add_argument('--normalize', dest='normalize', action='store_true')

    # args = parser.parse_args(extra_args)
    # TODO: add warning that other datasets not supported
    #args.dataset = "ptb"
    #language_modeling = Language_modeling(normalize=args.normalize,
    #                                      window_size=args.window_size,
    #                                      method=args.method,
    #                                      test=args.test)
    #language_modeling.run_with_args(args)
