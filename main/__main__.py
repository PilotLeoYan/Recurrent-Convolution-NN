import sys

from .utils.args import create_parser
from .utils.json_loader import load_config
from .utils.logger import get_logger, setup_logger


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    print(args)
    
    config = _config()

    setup_logger(config.get('logger', {}))
    logger = get_logger('RCNN')

    try:
        ...
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(e, exc_info=True)
        sys.exit(1)


def _config() -> dict:
    try:
        config = load_config(r'./config.json')
    except FileNotFoundError:
        print('"config.json" not found. Make sure to have this file in the root of the proyect.')
        sys.exit(1)
    except Exception as e:
        print(f'An exception during config loading: {e}')
        sys.exit(1)
    return config


if __name__ == '__main__':
    main()