import argparse
import torch
import matplotlib.pyplot as plt
from data import BirdsDataset
from argument import parse
from model import VGG16Transfer, Resnet18Transfer
import nntools as nt
from utils import ClassificationStatsManager, plot


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset
    train_set = BirdsDataset(args.root_dir, image_size=args.image_size)
    val_set = BirdsDataset(args.root_dir, mode='val',
                           image_size=args.image_size)
    num_classes = train_set.number_of_classes()

    # model
    if args.model == 'vgg':
        net = VGG16Transfer(num_classes).to(device)
    else:
        net = Resnet18Transfer(num_classes).to(device)

    # optimizer
    adam = torch.optim.Adam(net.parameters(), lr=args.lr)

    # stats manager
    stats_manager = ClassificationStatsManager()

    # experiment
    exp = nt.Experiment(net, train_set, val_set, adam, stats_manager, batch_size=args.batch_size,
                        output_dir=args.output_dir, perform_validation_during_training=True)

    # run
    if args.plot:
        fig, axes = plt.subplots(ncols=2, figsize=(7, 3))
        exp.run(num_epochs=args.num_epochs,
                plot=lambda exp: plot(exp, fig=fig, axes=axes))
    else:
        exp.run(num_epochs=args.num_epochs)


if __name__ == '__main__':
    args = parse()
    print(args)
    run(args)
