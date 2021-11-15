import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run model(NGCF,MF)")

    # save_weight, result path
    parser.add_argument('--weights_path', default='',
                        help='Save model path.')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Store model to path.')

    # data
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='Input data path.')
    parser.add_argument('--dataset', type=str, default='Movielens-100k', #Gowalla
                        help='Dataset name: { Amazon-book, Gowalla, Movielens-100k }')

    # hyperparameter
    parser.add_argument('--epochs', type=int, default=1, #400
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=16, # 1024 # 16
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, #0.0001
                        help='Learning rate.')
    parser.add_argument('--embed_dim', type=int, default=8, #
                        help='Embedding dimension.')
    parser.add_argument('--layers_output_size', type=str, default='[8,8,8,8]', #
                        help='Output sizes of every layer adjusting n_layers')

    # 보류 1 ( 1, 3 )
    parser.add_argument('--reg', type=float, default=1e-5, #
                        help='l2 regularization.')
    # parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
    #                     help='Regularizations.')

    # 보류 2 ( 1, 3 )
    parser.add_argument('--node_dropout', type=float, default=0.,
                        help='Node dropout. if you use this option, modify the ngcf forward function')
    # parser.add_argument('--mess_dropout', type=float, default=0.1,
    #                     help='Message dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1]', #0.1, 0.1
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    # data & model
    parser.add_argument('--adj_type', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from { plain, norm, mean }.')
    parser.add_argument('--model_type', default='lr_gccf', #ngcf
                        help='Specify the type of the graph convolutional layer from { ngcf, mf }.')

    # metric ( 1, 3 )
    parser.add_argument('--k', type=str, default=20,
                        help='k order of metric evaluation (e.g. NDCG@k, Recall@k)')
    #     parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
    #                         help='Output sizes of every layer')
    parser.add_argument('--eval_N', type=int, default=1, #10
                            help='Evaluate every N epochs')

    # save
    parser.add_argument('--save_results', type=int, default=1,
                        help='Save model and results')

    return parser.parse_args()

# 1
# def parse_args():
#
#     parser.add_argument('--node_dropout', type=float, default=0.,
#                         help='Graph Node dropout.')
#     parser.add_argument('--mess_dropout', type=float, default=0.1,
#                         help='Message dropout.')
#     parser.add_argument('--k', type=str, default=20,
#                         help='k order of metric evaluation (e.g. NDCG@k)')
#     parser.add_argument('--eval_N', type=int, default=1,
#                         help='Evaluate every N epochs')
#     parser.add_argument('--save_results', type=int, default=1,
#                         help='Save model and results')
#
#     return parser.parse_args()

# 3
# def parse_args():
#     parser.add_argument('--pretrain', type=int, default=0,
#                         help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
#     parser.add_argument('--verbose', type=int, default=1,
#                         help='Interval of evaluation.')
#
#     parser.add_argument('--gpu_id', type=int, default=0,
#                         help='0 for NAIS_prod, 1 for NAIS_concat')
#
#     parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
#                         help='Output sizes of every layer')
#
#     parser.add_argument('--save_flag', type=int, default=0,
#                         help='0: Disable model saver, 1: Activate model saver')
#
#     parser.add_argument('--test_flag', nargs='?', default='part',
#                         help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
#
#     parser.add_argument('--report', type=int, default=0,
#                         help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')