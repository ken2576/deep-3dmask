import configargparse

def get_opts():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument('--root_dir', type=str,
                        help='root directory of dataset')
    parser.add_argument('--val_dir', type=str,
                        help='root directory of validation dataset')
    
    parser.add_argument('--dataset_name', type=str, default='realestate',
                        choices=['realestate'],
                        help='which dataset to train/val')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for all RNGs')
    
    parser.add_argument('--img_wh', type=int, default=[640, 360],
                        nargs='+',
                        help='input image dimensions')

    parser.add_argument('--loss_type', type=str, default='l1',
                        choices=['mse', 'l1', 'vgg'],
                        help='which loss to use')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss', 'vgg'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=1.0,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='path to log files')
    parser.add_argument('--ckpt_dir', type=str, default='ckpts',
                        help='path to checkpoints')

    parser.add_argument('--debug', action='store_true',
                        help='enable debug mode')

    return parser.parse_args()
