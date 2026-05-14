import sys

from .evaluate import eval_models
from .train.train import train_models
from .visualize import visualize_inference
from .utils.args import create_parser
from .utils.json_loader import load_config
from .utils.logger import get_logger, setup_logger


def main() -> None:
    parser = create_parser()
    args = vars(parser.parse_args())

    if args.get("config", None) is not None:
        config = _config(args["config"])
    else:
        config = _config()

    setup_logger(config.get("logger", {}))
    logger = get_logger("RCNN")

    logger.info("Running __main__.py")

    try:
        if args.get("fit", False):
            train_models(args=args["fit"], config=config.get("fit", {}))
        elif args.get("test", False):
            eval_models(config=config.get("eval", {}))
        elif args.get("visualize", False):
            visualize_inference(config=config.get("visualize", {}))
        else:
            print('No parameter was selected. Use "python -m main -h" for help.')

    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(e, exc_info=True)
        sys.exit(1)

    logger.info("__main__.py finished.\n")


def _config(path: str = r"./config.json") -> dict:
    try:
        config = load_config(path)
    except FileNotFoundError:
        print(
            '"config.json" not found. Make sure to have this file in the root of the proyect.'
        )
        sys.exit(1)
    except Exception as e:
        print(f"An exception during config loading: {e}")
        sys.exit(1)
    return config


if __name__ == "__main__":
    main()
