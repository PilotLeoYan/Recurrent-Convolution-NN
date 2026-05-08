import torch

try:
    from utils.logger import get_logger
    from data import download_moving_mnist
    from data import MovingMNISTDataset
    from data import augmentation
    from data import make_dataloader

    from models import RCNN2d
    from losses import get_loss_fn
    from optimizers import get_optimizer
except ModuleNotFoundError:
    from ..utils.logger import get_logger
    from ..data import download_moving_mnist
    from ..data import MovingMNISTDataset
    from ..data import augmentation
    from ..data import make_dataloader

    from ..models import RCNN2d
    from ..losses import get_loss_fn
    from ..optimizers import get_optimizer

logger = get_logger(__name__)


def train_models(
    args: str,
    config: dict,
):
    """
    """
    if len(config) == 0: # if config is empty
        _config = {
            'hidden_channels': 1,
            'kernel_size': 3,
            'units': 8,
            'activation': 'relu',
            'device': 'cpu',
            'lr': 0.001,
            'weight_decay': 0.01,
            'split_seq_len': 10,
            'epochs': 2,
            'train': {
                'batch_size': 16,
                'data_augmentation': True,
                'n_workers': 0,
            },
            'valid': {
                'batch_size': 32,
                'n_workers': 0,
            },
            'test': {
                'batch_size': 32,
                'n_workers': 0,
            },
        }
        logger.warning('Not found "fit" configuration in config.json. Please add this template to config.json: %s', _config)
        return
    logger.info('"fit" configuration received: %s', config)

    dataloaders = _load_dataloader(config)

    batch = next(iter(dataloaders[0])) # (# samples, seq_len, Channels_in, H, W)

    models: list[torch.nn.Module] = []
    
    match(args):
        case 'rcnn' | 'all':
            
            models.append(RCNN2d(
                input_channels=batch.shape[2],
                hidden_channels=config['hidden_channels'],
                kernel_size=config['kernel_size'],
                units=config['units'],
                activation=config['activation'],
            ))
        case _:
            logger.warning('No available models selected to train.')
            return

    _train_models(models, dataloaders[0], dataloaders[1], config)


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
        train=True,
        n_workers=config['train']['n_workers'],
    )

    valid_dl = make_dataloader(
        valid_ds, 
        batch_size=config['valid']['batch_size'],
        n_workers=config['valid']['n_workers'],
    )

    test_dl = make_dataloader(
        test_ds, 
        batch_size=config['test']['batch_size'],
        n_workers=config['test']['n_workers'],
    )

    return train_dl, valid_dl, test_dl


def _train_models(
    models: list[torch.nn.Module], 
    train_loader,
    valid_loader,
    config: dict
):
    """
    """
    for model in models:
        model.to(config['device'])
        
        loss_fn = get_loss_fn()
        optimizer = get_optimizer(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        for epoch in range(config['epochs']):
            logger.info('Starting epoch %i', epoch + 1)

            model.train(True)
            avg_loss = _train_one_epoch(
                train_loader, optimizer, model, loss_fn, config
            )

            running_vlos = 0.0
            model.eval()
            with torch.no_grad():
                for i, vbatch in enumerate(valid_loader):
                    vinputs, vlabels = _transpose_data(vbatch, config['split_seq_len'])
                    vinputs = vinputs.to(config['device'])
                    vlabels = vlabels.to(config['device'])

                    voutputs = model(vinputs)
                    vloss = loss_fn(voutputs, vlabels)
                    running_vlos += vloss
            avg_vloss = running_vlos / (i + 1)

            logger.info('Loss: %i, Loss_v: %i', avg_loss, avg_vloss)


def _train_one_epoch(
    training_loader,
    optimizer,
    model,
    loss_fn,
    config: dict,
) -> float:
    """
    """
    running_loss = 0.0

    for i, batch in enumerate(training_loader):
        inputs, labels = _transpose_data(batch, config['split_seq_len'])
        inputs = inputs.to(config['device'])
        labels = labels.to(config['device'])

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / (i + 1)

def _transpose_data(
    batch: torch.Tensor,
    split: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns inputs, labels
    """
    # from (# samples, seq_len, C_in, H, W) -> (seq_len, # samples, C_in, H, W)
    batch = batch.clone().transpose(0, 1)
    return batch[:split], batch[split:]