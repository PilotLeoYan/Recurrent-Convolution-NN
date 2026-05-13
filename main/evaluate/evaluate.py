import torch

try:
    from data import MovingMNISTDataset, download_moving_mnist, make_dataloader
    from models import RCNN2d, Conv2dGRU, CNN, transpose_data
    from train import get_prediction
    from utils.logger import get_logger
except ModuleNotFoundError:
    from ..data import MovingMNISTDataset, download_moving_mnist, make_dataloader
    from ..models import RCNN2d, Conv2dGRU, CNN, transpose_data
    from ..train import get_prediction
    from ..utils.logger import get_logger
from .metrics import metrics

logger = get_logger(__name__)

# Maps the config key -> (ModelClass, model.name)
_MODEL_REGISTRY: dict[str, tuple[type, str]] = {
    "rcnn2d": (RCNN2d, "rcnn2d"),
    "cgru": (Conv2dGRU, "cgru"),
    "cnn":  (CNN, "cnn"),
}


def eval_models(config: dict) -> None:
    """
    """
    if len(config) == 0:
        logger.warning('Not found "eval" configuration in config.json.')
        return

    model_type = config.get("model_type", "not selected").lower()

    if model_type in _MODEL_REGISTRY:
        model_cls, model_name = _MODEL_REGISTRY[model_type]
    else:
        logger.error(
            'Unknown model_type "%s". Valid options: %s.',
            model_type,
            list(_MODEL_REGISTRY.keys()),
        )
        return

    test_loader = _load_test(config)
    first_batch = next(iter(test_loader))

    model_path = config["model_path"]

    logger.info('Evaluating model "%s" from "%s"', model_name, model_path)

    model, loss_fn = _load_model(
        batch=first_batch,
        config=config,
        model_cls=model_cls,
        model_path=model_path,
    )
    model.to(config["device"])
    model.eval()

    running_loss = 0.0
    running_metrics = metrics(torch.zeros((1, )), torch.zeros(1, )) # type: ignore
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            with torch.no_grad():
                inputs, labels = transpose_data(batch, config["split_seq_len"])
                inputs = inputs.to(config["device"])
                labels = labels.to(config["device"])

                predictions = get_prediction(
                    model, inputs, labels, teacher_forcing_ratio=0.0,
                )

                loss = loss_fn(predictions, labels)
                running_loss += loss.item()

                metrics_values = metrics(predictions, labels)
                for key in metrics_values:
                    running_metrics[key] += metrics_values[key]

    avg_loss = running_loss / (i + 1) # type: ignore[possibly-undefined]
    logger.info('Test [%s], loss: %f', model_name, avg_loss)

    avg_metrics = metrics(torch.zeros((1, )), torch.zeros(1, )) # type: ignore
    for key in running_metrics:
        avg_metrics[key] = running_metrics[key] / (i + 1) # type: ignore[possibly-undefined]
        logger.info('Test [%s] metrics, %s: %f', model_name, key.upper(), avg_metrics[key])

# Helpers
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
    batch: torch.Tensor,
    config: dict,
    model_cls: type,
    model_path: str,
) -> tuple[torch.nn.Module, torch.nn.MSELoss]:
    """
    Instantiate *model_cls* with the hyper-parameters from *config* and
    load the checkpoint at *model_path*.

    ``Conv2dGRU`` and ``CNN`` do not accept an ``activation`` kwarg, so
    that argument is only forwarded to ``RCNN2d``.
    """
    kwargs: dict = dict(
        input_channels=batch.shape[2],
        hidden_channels=config["hidden_channels"],
        kernel_size=config["kernel_size"],
        units=config["units"],
    )
    if model_cls is RCNN2d:
        kwargs["activation"] = config.get("activation", "relu")

    model = model_cls(**kwargs)

    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = checkpoint["loss"]

    return model, loss_fn
