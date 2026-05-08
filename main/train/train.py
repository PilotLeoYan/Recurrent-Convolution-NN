try:
    from utils.logger import get_logger
    from data import download_moving_mnist
    from data import MovingMNISTDataset
    from data import augmentation
    from data import make_dataloader
except ModuleNotFoundError:
    from ..utils.logger import get_logger
    from ..data import download_moving_mnist
    from ..data import MovingMNISTDataset
    from ..data import augmentation
    from ..data import make_dataloader

logger = get_logger(__name__)


def train_models(
    args: str,
    config: dict,
):
    """
    """
    if len(config) == 0: # if config is empty
        # default config
        config = {
            'train': {
                'batch_size': 16,
                'data_augmentation': True
            },
            'valid': {
                'batch_size': 32,
            },
            'test': {
                'batch_size': 32,
            },
        }
        logger.warning('Not found "fit" configuration in config.json. Loading default configuration: %s', config)

    dataloaders = _load_dataloader(config)

    for loader in dataloaders:
        batch = next(iter(loader))
        print(batch.shape)

    return

    match(args):
        case 'rcnn':
            from models.rcnn import RCNN2d
            model = RCNN2d()

    _train_model(model, config)


def _load_dataloader(
    config: dict
) -> tuple:
    """
    """
    paths = download_moving_mnist()

    aug = None
    # if config.get('train', {}).get('data_augmentation', False):
    if config['train']['data_augmentation']:
        aug = augmentation()
    
    train_ds = MovingMNISTDataset(paths[0], transform=aug)
    valid_ds = MovingMNISTDataset(paths[1])
    test_ds = MovingMNISTDataset(paths[2])

    train_dl = make_dataloader(
        train_ds, 
        batch_size=config['train']['batch_size'],
    )

    valid_dl = make_dataloader(
        valid_ds, 
        batch_size=config['valid']['batch_size'],
    )

    test_dl = make_dataloader(
        test_ds, 
        batch_size=config['test']['batch_size'],
    )

    return train_dl, valid_dl, test_dl


def _train_model():
    ...