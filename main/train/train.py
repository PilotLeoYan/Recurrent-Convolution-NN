import torch

try:
    from data import (
        MovingMNISTDataset,
        augmentation,
        download_moving_mnist,
        make_dataloader,
    )
    from losses import get_loss_fn
    from models import (
        RCNN2d,
        predict_rcnn2d,
        transpose_data,
        Conv2dGRU,
        predict_cgru,
        CNN,
        predict_cnn,
    )
    from optimizers import get_optimizer, get_lr_scheduler
    from utils.logger import get_logger
    from utils.csv_logger import CSVTrainingLogger
except ModuleNotFoundError:
    from ..data import (
        MovingMNISTDataset,
        augmentation,
        download_moving_mnist,
        make_dataloader,
    )
    from ..losses import get_loss_fn
    from ..models import (
        RCNN2d,
        predict_rcnn2d,
        transpose_data,
        Conv2dGRU,
        predict_cgru,
        CNN,
        predict_cnn,
    )
    from ..optimizers import get_optimizer, get_lr_scheduler
    from ..utils.logger import get_logger
    from ..utils.csv_logger import CSVTrainingLogger

logger = get_logger(__name__)


def train_models(
    args: str,
    config: dict,
):
    """ """
    if len(config) == 0:  # if config is empty
        _config = {
            "hidden_channels": 1,
            "kernel_size": 3,
            "units": 8,
            "activation": "relu",
            "device": "cpu",
            "lr": 0.001,
            "weight_decay": 0.01,
            "split_seq_len": 10,
            "epochs": 2,
            "train": {
                "batch_size": 16,
                "data_augmentation": True,
                "n_workers": 0,
            },
            "valid": {
                "batch_size": 32,
                "n_workers": 0,
            },
            "test": {
                "batch_size": 32,
                "n_workers": 0,
            },
        }
        logger.warning(
            'Not found "fit" configuration in config.json. Please add this template to config.json: %s',
            _config,
        )
        return
    logger.info('"fit" configuration received: %s', config)

    dataloaders = _load_dataloader(config)

    batch = next(iter(dataloaders[0]))  # (# samples, seq_len, Channels_in, H, W)

    models: list[torch.nn.Module] = []

    if args == 'rcnn' or args == 'all':
        models.append(
            RCNN2d(
                input_channels=batch.shape[2],
                hidden_channels=config["hidden_channels"],
                kernel_size=config["kernel_size"],
                units=config["units"],
                activation=config["activation"],
            )
        )
    if args == 'cgru' or args == 'all':
        models.append(
            Conv2dGRU(
                input_channels=batch.shape[2],
                hidden_channels=config["hidden_channels"],
                kernel_size=config["kernel_size"],
                units=config["units"],
            )
        )
    if args == 'cnn' or args == 'all':
        models.append(
            CNN(
                input_channels=batch.shape[2],
                hidden_channels=config["hidden_channels"],
                kernel_size=config["kernel_size"],
                units=config["units"],
            )
        )

    if len(models) == 0:
        logger.warning("No available models selected to train.")
        return

    _train_models(models, dataloaders[0], dataloaders[1], config)


def _load_dataloader(config: dict) -> tuple:
    """ """
    paths = download_moving_mnist()

    aug = None
    if config["train"]["data_augmentation"]:
        aug = augmentation()

    train_ds = MovingMNISTDataset(paths[0], transform=aug)
    valid_ds = MovingMNISTDataset(paths[1])

    train_dl = make_dataloader(
        train_ds,
        batch_size=config["train"]["batch_size"],
        train=True,
        n_workers=config["train"]["n_workers"],
    )

    valid_dl = make_dataloader(
        valid_ds,
        batch_size=config["valid"]["batch_size"],
        n_workers=config["valid"]["n_workers"],
    )

    return train_dl, valid_dl


def _train_models(
    models: list[torch.nn.Module], train_loader, valid_loader, config: dict
):
    """
    Trains each model and records per-epoch metrics to a CSV file inside
    the folder defined by config["csv_log"]["output_dir"]
    (default: "saves/training_logs").
    """
    csv_cfg = config.get("csv_log", {})
    csv_output_dir: str = csv_cfg.get("output_dir", "saves/training_logs")

    lr_sched_cfg = config.get('lr_scheduler', {})

    sched_type   = lr_sched_cfg.get("type", "cosine")

    for model in models:
        logger.info('Init training of %s', model.name)

        best_vloss = torch.inf
        best_epoch = -1

        model.to(config["device"])

        loss_fn = get_loss_fn()
        optimizer = get_optimizer(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )

        lr_scheduler = get_lr_scheduler(
            optimizer,
            scheduler_type=sched_type,
            step_size=lr_sched_cfg.get("step_size", 3),
            gamma=lr_sched_cfg.get("gamma", 0.5),
            T_max=lr_sched_cfg.get("T_max", config["epochs"]),
            eta_min=lr_sched_cfg.get("eta_min", 1e-6),
        )

        running_vlos = [0.0, 0.0]
        model.eval()
        with torch.no_grad():
            for j, loader in enumerate((train_loader, valid_loader)):
                for i, vbatch in enumerate(loader):
                    vinputs, vlabels = transpose_data(vbatch, config["split_seq_len"])
                    vinputs = vinputs.to(config["device"])
                    vlabels = vlabels.to(config["device"])

                    vpredictions = get_prediction(model, vinputs, vlabels, 0.0)

                    vloss = loss_fn(vpredictions, vlabels)
                    running_vlos[j] += vloss.item()
                running_vlos[j] = running_vlos[j] / (i + 1)  # type: ignore
            logger.info("Init Loss: %f, Loss_v: %f", running_vlos[0], running_vlos[1])

        # ── CSV logger: one file per model per run ─────────────────────
        with CSVTrainingLogger(
            output_dir=csv_output_dir,
            model_name=model.name, # type: ignore
        ) as csv_log:

            csv_log.log(
                epoch=0,
                train_loss=running_vlos[0],
                val_loss=running_vlos[1],
                teacher_forcing_ratio=-1,
                learning_rate=-1,
                is_best=False,
            )

            for epoch in range(config["epochs"]):
                logger.info("Starting epoch %i", epoch + 1)

                tf_ratio = max(0.05, 1.0 - epoch / max(config['epochs'] - 1, 1)) # lineal decay

                model.train(True)
                avg_loss = _train_one_epoch(
                    train_loader, optimizer, model,
                    loss_fn, tf_ratio, config)

                running_vlos = 0.0
                model.eval()
                with torch.no_grad():
                    for i, vbatch in enumerate(valid_loader):
                        vinputs, vlabels = transpose_data(vbatch, config["split_seq_len"])
                        vinputs = vinputs.to(config["device"])
                        vlabels = vlabels.to(config["device"])

                        vpredictions = get_prediction(model, vinputs, vlabels, 0.0)

                        vloss = loss_fn(vpredictions, vlabels)
                        running_vlos += vloss.item()
                avg_vloss = running_vlos / (i + 1)  # type: ignore

                # use .step() always after of valid step.
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(avg_vloss)
                else:
                    lr_scheduler.step()

                # Determine whether this epoch is the new best
                is_best = config['save_best'] and avg_vloss < best_vloss

                if is_best:
                    best_vloss = avg_vloss
                    best_epoch = epoch + 1
                    try:
                        torch.save({
                            'epoch': best_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss_fn,  # type: ignore
                        }, config['model_path'] + f'_{model.name}_best.pth')
                    except FileNotFoundError:
                        from pathlib import Path
                        Path(config['model_path']).parent.mkdir(parents=True, exist_ok=True)

                        torch.save({
                            'epoch': best_epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss_fn,  # type: ignore
                        }, config['model_path'] + f'_{model.name}_best.pth')

                # ── Write one CSV row for this epoch ───────────────────
                csv_log.log(
                    epoch=epoch + 1,
                    train_loss=avg_loss,
                    val_loss=float(avg_vloss),
                    teacher_forcing_ratio=tf_ratio,
                    learning_rate=optimizer.param_groups[0]['lr'],
                    is_best=is_best,
                )

                logger.info("Loss: %f, Loss_v: %f", avg_loss, avg_vloss)

        # ── Post-training summary / final checkpoint ───────────────────
        if config['save_best']:
            logger.info("Best epoch at %i, Loss_v: %f", best_epoch, best_vloss)
        else:
            from pathlib import Path
            Path(config['model_path']).parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch + 1,  # type: ignore
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn,  # type: ignore
            }, config['model_path'] + f'_{model.name}_last.pth')


def _train_one_epoch(
    training_loader,
    optimizer,
    model,
    loss_fn,
    teacher_forcing_ratio: float,
    config: dict,
) -> float:
    """ """
    running_loss = 0.0

    for i, batch in enumerate(training_loader):
        inputs, labels = transpose_data(batch, config["split_seq_len"])
        inputs = inputs.to(config["device"])
        labels = labels.to(config["device"])

        optimizer.zero_grad()

        predictions = get_prediction(model, inputs, labels, teacher_forcing_ratio)

        loss = loss_fn(predictions, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # gradient clipping
        optimizer.step()
        running_loss += loss.item()

    return running_loss / (i + 1) # type: ignore


def get_prediction(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    teacher_forcing_ratio: float,
) -> torch.Tensor:
    """ """
    if isinstance(model, RCNN2d):
        predictions = predict_rcnn2d(
            model, # type: ignore
            inputs,
            labels,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
    elif isinstance(model, Conv2dGRU):
        predictions = predict_cgru(
            model, # type: ignore
            inputs,
            labels,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
    elif isinstance(model, CNN):
        predictions = predict_cnn(
            model,  # type: ignore
            inputs,
            labels,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
    else:
        raise TypeError(
            f"No predict function available for model type '{type(model).__name__}'. "
            f"Expected one of: RCNN2d, Conv2dGRU, CNN."
        )

    return predictions  # (pred_len, batch, 1, H, W)
