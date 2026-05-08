import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='List of all arguments:'
    )

    group = parser.add_mutually_exclusive_group()

    # not used yet
    group.add_argument('-t', '--test', help='init all the testing.',
        type=str,
        choices=('all', 'rcnn', 'other'),
        default=None)

    group.add_argument('-f', '--fit', help='init the fitting.',
        type=str,
        choices=('all', 'rcnn', 'other'),
        default=None)

    group.add_argument('-ft', '--fit_test', help='init the fitting and then testing.',
        type=str,
        choices=('all', 'rcnn', 'other'),
        default=None)

    return parser