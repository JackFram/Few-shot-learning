import argparse

from train import main

parser = argparse.ArgumentParser(description='xxx')

# data args
default_dataset = 'omniglot'
parser.add_argument('--data.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
default_split = 'vinyals'
parser.add_argument('--data.split', type=str, default=default_split, metavar='SP',
                    help="split name (default: {:s})".format(default_split))
parser.add_argument('--data.way', type=int, default=60, metavar='WAY',
                    help="number of classes per episode (default: 60)")
parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=5, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.train_episodes', type=int, default=100, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=100, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")
parser.add_argument('--data.trainval', action='store_true', help="run in train+validation mode (default: False)")
parser.add_argument('--data.sequential', action='store_true', help="use sequential sampler instead of episodic (default: False)")
parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")

args = vars(parser.parse_args())

main(args)

print(main(args))