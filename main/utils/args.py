import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='List of all arguments:'
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument('-t', '--test', help='init the testing of one model.',
        action='store_true')

    group.add_argument('-f', '--fit', help='init the fitting.',
        type=str,
        choices=('all', 'rcnn', 'cgru', 'cnn'),
        default=None)

    group.add_argument('-v', '--visualize', help='generate inference images from a pretrained model.',
        action='store_true')

    parser.add_argument('-c', '--config', help='pass a diferent config.json path.',
        type=str)

    return parser
