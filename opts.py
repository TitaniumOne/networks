import argparse
parser = argparse.ArgumentParser(
    description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset',
                    type=str,
                    choices=['ucf101', 'diving48', 'something-v1', 'skating2'])
parser.add_argument('modality', type=str, choices=['RGB', 'RGBDiff', 'flow'])
parser.add_argument('--lr_scheduler',
                    dest='lr_scheduler',
                    type=str,
                    choices=['cos_warmup', 'lr_step_warmup', 'lr_step'],
                    default='cos_warmup')
# ========================= Model Configs ==========================
parser.add_argument('--sample_frames', type=int, default=128)
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--dropout',
                    '--do',
                    default=0.8,
                    type=float,
                    metavar='DO',
                    help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll", choices=['nll'])
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--alpha',
                    type=int,
                    default=4,
                    help='spatial temporal split for output channels')
parser.add_argument(
    '--beta',
    type=int,
    default=2,
    choices=[1, 2],
    help='channel splits for input channels, 1 for GST-Large and 2 for GST')
parser.add_argument('--resi',
                    dest='resi',
                    action='store_true',
                    help='open residual connection or not',
                    default=False)
parser.add_argument('--partial_bn',
                    dest='partial_bn',
                    action='store_true',
                    help='open residual connection or not, only for TSN',
                    default=False)
# ========================= Learning Configs ==========================
parser.add_argument('--epochs',
                    default=120,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b',
                    '--batch-size',
                    default=256,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.01,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--lr_steps',
                    default=[50, 60],
                    type=float,
                    nargs="+",
                    metavar='LRSteps',
                    help='epochs to decay learning rate by 10')
parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient',
                    '--gd',
                    default=None,
                    type=float,
                    metavar='W',
                    help='gradient norm clipping (default: disabled)')
parser.add_argument('--num_segments', type=int, default=7)
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument(
    '--iter_size',
    default=2,
    type=int,
    help='Number of iterations to wait before updating the weights')
parser.add_argument('--warmup',
                    default=10,
                    type=int,
                    help='Number of epochs for linear warmup')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq',
                    '-p',
                    default=20,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--eval-freq',
                    '-ef',
                    default=1,
                    type=int,
                    metavar='N',
                    help='evaluation frequency (default: 5)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume_rgb',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--rgb_weights_path',
                    default='model/model_rgb.pth',
                    type=str,
                    help='path to latest checkpoint ')
parser.add_argument('-e',
                    '--evaluate',
                    dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")  # ???
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--run_iter', type=int, default=1)
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='model')
parser.add_argument('--root_output', type=str, default='output')
