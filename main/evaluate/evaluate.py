import torch

try:
    from data import MovingMNISTDataset, download_moving_mnist, make_dataloader
    from models import RCNN2d, predict_rcnn2d, transpose_data
    from utils.logger import get_logger
except ModuleNotFoundError:
    from ..data import MovingMNISTDataset, download_moving_mnist, make_dataloader
    from ..models import RCNN2d, predict_rcnn2d, transpose_data
    from ..utils.logger import get_logger

logger = get_logger(__name__)


def eval_models(config: dict) -> None:
    """ """
    if len(config) == 0:
        logger.warning('Not found "eval" configuration in config.json.')
        return

    test_loader = _load_test(config)

    logger.info('Loading model from "%s"', config['model_path'])
    model, loss_fn = _load_model(next(iter(test_loader)), config)
    model.to(config["device"])
    model.eval()

    running_loss = 0.0
    for i, batch in enumerate(test_loader):
        with torch.no_grad():
            inputs, labels = transpose_data(batch, config["split_seq_len"])
            inputs = inputs.to(config["device"])
            labels = labels.to(config["device"])

            predictions = predict_rcnn2d(
                model,  # type: ignore
                inputs,
                labels,
                teacher_forcing_ratio=0.0,
            )

            loss = loss_fn(predictions, labels)
            running_loss += loss
    avg_loss = running_loss / (i + 1)  # type: ignore
    logger.info("Test loss: %f", avg_loss)


def _load_test(config: dict) -> torch.utils.data.DataLoader:
    """ """
    paths = download_moving_mnist()

    test_ds = MovingMNISTDataset(paths[2])

    return make_dataloader(
        test_ds,
        batch_size=config["batch_size"],
        n_workers=config["n_workers"],
    )


def _load_model(
    batch: torch.Tensor, config: dict
) -> tuple[torch.nn.Module, torch.nn.MSELoss]:
    """ """
    model = RCNN2d(
        input_channels=batch.shape[2],
        hidden_channels=config["hidden_channels"],
        kernel_size=config["kernel_size"],
        units=config["units"],
        activation=config["activation"],
    )
    checkpoint = torch.load(config["model_path"], weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss = checkpoint["loss"]

    return model, loss
