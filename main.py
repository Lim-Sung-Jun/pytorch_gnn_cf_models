from model.model_wrapper import *
from utility.parser import *
from utility.load_data import *

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print("available device: ", device)

if __name__ == '__main__':

    # parser
    args = parse_args()
    print(args)
    print()

    # dataloader
    data_generator = Data(path=args.data_path + args.dataset, batch_size=args.batch_size)
    plain_adj, lr_gccf_adj, ngcf_adj = data_generator.get_adj_mat()

    # data_config, args
    data_config = {}
    data_config['n_users'], data_config['n_items']  = data_generator.n_users, data_generator.n_items

    if args.model_type == 'lr_gccf':
        # D^(-1/2)(A + I)D^(-1/2)
        data_config['norm_adj'] = lr_gccf_adj
        print('use the normalized adjacency matrix')
    elif args.model_type == 'ngcf':
        # D^-1(A + I)
        data_config['norm_adj'] = ngcf_adj
        print('use the mean adjacency matrix')
    else: # A
        data_config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')

    # model
    Engine = Model_Wrapper(data_config=data_config, args = args, data_generator = data_generator)

    # model train
    Engine.train()

    print('end')