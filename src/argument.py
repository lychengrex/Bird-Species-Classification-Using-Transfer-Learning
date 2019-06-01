import argparse


def parse():
    '''
    Add arguments.
    '''
    parser = argparse.ArgumentParser(
        description='Bird-Species-Classification-Using-Transfer-Learning')

    parser.add_argument('--root_dir', type=str,
                        default='../dataset/', help='root directory of dataset')
    parser.add_argument('--output_dir', type=str,
                        default='../checkpoints/', help='directory of saved checkpoints')
    parser.add_argument('--num_epochs', type=int,
                        default=20, help='number of epochs')
    parser.add_argument('--plot', type=bool, default=False,
                        help='plot loss during training or not')
    parser.add_argument('--model', type=str, default='vgg',
                        help='vgg or resnet')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for training')
    parser.add_argument('--image_size', type=tuple, default=(224, 224))
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--train', action='store_true',
                        help='whether train DQN')
    parser.add_argument('--test', action='store_true', help='whether test DQN')

    return parser.parse_args()


class Args():
    '''
    For jupyter notebook
    '''

    def __init__(self):
        self.root_dir = '../dataset/'
        self.output_dir = '../checkpoints/'
        self.num_epochs = 20
        self.plot = False
        self.model = 'vgg'
        self.lr = 1e-3
        self.image_size = (224, 224)
        self.batch_size = 16
