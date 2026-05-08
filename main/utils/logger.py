import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape


class ConsoleMarkupFormatter(logging.Formatter):
    """
    Custom formatter that injects rich color tags based on the log level, 
    respecting the exact structure desired.
    """
    LEVEL_COLORS = {
        "DEBUG": "cyan",
        "INFO": "bold green",
        "WARNING": "bold yellow",
        "ERROR": "bold red",
        "CRITICAL": "bold white on red",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "white")
        asctime = self.formatTime(record, self.datefmt)
        message = escape(record.getMessage())
        
        return (
            f"[dim cyan]{asctime}[/dim cyan] | "
            f"[{color}][{record.levelname}][/{color}] | "
            f"[magenta]{record.name}:{record.funcName}[/magenta] - "
            f"{message}"
        )


def _setup_logger(
    level: str,
    log_to_file: bool,
    log_file: str,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    
    _console = Console(stderr=True)

    console_handler = RichHandler(
        console=_console,
        show_time=False,
        show_level=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    
    console_formatter = ConsoleMarkupFormatter(datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_formatter)

    handlers: list[logging.Handler] = [console_handler]

    if log_to_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | [%(levelname)s] | %(name)s:%(funcName)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handlers.append(file_handler)

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True, 
    )


def setup_logger(config: dict) -> None:
    _params = { # default arguments
        'level': 'INFO', 
        'log_to_file': True, 
        'log_file': 'logs/logs.log', 
        'max_bytes': 5242880,
        'backup_count': 3
    }

    if not config: # if it's empty
        print('[!] not found "logger" config in config.json. Setting default.')
        _setup_logger(**_params)
        return

    for key in _params:
        try:
            _params[key] = config[key]
        except Exception as e:
            print(f'[!] not found logger.{key} in config.json. Setting default: {_params[key]}')

    _setup_logger(**_params)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
