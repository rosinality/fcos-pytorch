import argparse


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=0.0001)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=24)
    parser.add_argument('--n_save_sample', type=int, default=5)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('path', type=str)

    return parser


def get_args():
    parser = get_argparser()
    args = parser.parse_args()

    args.feat_channels = [0, 0, 512, 768, 1024]
    args.out_channel = 256
    args.use_p5 = True
    args.n_class = 81
    args.n_conv = 4
    args.prior = 0.01
    args.threshold = 0.05
    args.top_n = 1000
    args.nms_threshold = 0.6
    args.post_top_n = 100
    args.min_size = 0
    args.fpn_strides = [8, 16, 32, 64, 128]
    args.gamma = 2.0
    args.alpha = 0.25
    args.sizes = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 100000000]]
    args.train_min_size_range = (640, 800)
    args.train_max_size = 1333
    args.test_min_size = 800
    args.test_max_size = 1333
    args.pixel_mean = [0.40789654, 0.44719302, 0.47026115]
    args.pixel_std = [0.28863828, 0.27408164, 0.27809835]
    args.size_divisible = 32
    args.center_sample = True
    args.pos_radius = 1.5
    args.iou_loss_type = 'giou'

    return args
