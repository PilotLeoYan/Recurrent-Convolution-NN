import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Columns written to every CSV file
_FIELDNAMES = [
    "epoch",
    "train_loss",
    "val_loss",
    "teacher_forcing_ratio",
    "learning_rate",
    "is_best",
]


class CSVTrainingLogger:
    """
    Writes one row per epoch to a CSV file inside a dedicated folder.

    File naming:  <output_dir>/<model_name>_<run_timestamp>.csv

    Usage
    -----
    csv_log = CSVTrainingLogger(output_dir="saves/training_logs", model_name="rcnn")
    csv_log.log(epoch=1, train_loss=0.42, val_loss=0.38,
                teacher_forcing_ratio=0.9, learning_rate=0.0015, is_best=True)
    csv_log.close()
    """

    def __init__(self, output_dir: str, model_name: str) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = self._output_dir / f"{model_name}_{timestamp}.csv"

        self._file = self._path.open(mode="w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=_FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

        logger.info("CSV training log -> %s", self._path)

    # ------------------------------------------------------------------
    def log(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        teacher_forcing_ratio: float,
        learning_rate: float,
        is_best: bool = False,
    ) -> None:
        """Append one row to the CSV (one call per epoch)."""
        self._writer.writerow(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 6),
                "val_loss": round(float(val_loss), 6),
                "teacher_forcing_ratio": round(float(teacher_forcing_ratio), 4),
                "learning_rate": learning_rate,
                "is_best": is_best,
            }
        )
        self._file.flush()  # write immediately so the file is usable mid-run

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._file.closed:
            self._file.flush()
            self._file.close()
            logger.info("CSV log closed -> %s", self._path)

    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        """Absolute path to the CSV file being written."""
        return self._path.resolve()

    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
