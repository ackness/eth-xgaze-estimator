import argparse
from functools import partial


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class TrainConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Train Config")
        parser.add_argument('--mode', type=str, default='train')

        ########### Basic Settings ###########
        parser.add_argument('--model_name', type=str, default='timm_resnet50',
                            help='which model your wanna use, define in models/')
        parser.add_argument('--data_dir', type=str, default='',
                            help='path to training or test set')
        parser.add_argument('--data_type', type=str, default='with-in', choices=['with-in', 'cross'],
                            help='choice which Phase')

        ########### Cross Datasets Extra Settings (only valid if data_type==cross) ###########
        parser.add_argument('--xgaze_data_dir', type=str, default='',
                            help='path to xgaze dataset, if you want to train a cross dataset model, '
                                 'then use this path to xgaze to do val or test')

        ########### Training Settings ###########
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='LR')
        parser.add_argument('-op', '--optimizer', type=str, default='adam', help='optimizer')
        parser.add_argument('-bs', '--train_batch_size', type=int, default=50, help='batch size')
        parser.add_argument('-nw', '--num_workers', type=int, default=4, help='num of workers')
        parser.add_argument('-val', '--use_val', action='store_true',
                            help='whether to split a part of train set to val set')
        parser.add_argument('-val_bs', '--val_batch_size', type=int, default=50, help='val loader batch size')
        parser.add_argument('-sr', '--split_ratio', type=float, default=0.9,
                            help='split train set ratio if use val set')
        parser.add_argument('-pose', '--is_load_pose', action='store_true',
                            help='whether to use pose label in dataset')
        parser.add_argument('--use_aa', action='store_true',
                            help='whether to use augment policy (define in datasets/transforms_policy) '
                                 'in training process. if False will use default transforms')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))


class TestConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Test Config")
        parser.add_argument('--mode', type=str, default='test')

        parser.add_argument('--model_name', type=str, default='')
        parser.add_argument('--ckpt_path', type=str, default='')
        parser.add_argument('--is_load_pose', action='store_true')
        parser.add_argument('-bs', '--test_batch_size', type=int, default=512)
        parser.add_argument('-nw', '--num_workers', type=int, default=8)

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))
