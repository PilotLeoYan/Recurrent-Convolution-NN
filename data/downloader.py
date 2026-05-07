import hashlib
from pathlib import Path
import shutil
import requests
from tqdm import tqdm
from typing import Iterator
import time
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


CHUNK_SIZE: int = 8 * 1024 * 1024 # 8 MiB
MAX_RETRIES: int = 3
RETRY_BACKOFF: float = 2.0 # seconds
REQUEST_TIMEOUT: tuple[int, int] = (10, 60) # (connect, read) seconds


MOVING_MINST_URL: str = ('https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy')
MOVING_MINST_SHA256: str | None = None


def download_moving_mnist(
    data_path: str | Path = r'dataset',
    train: float = 0.7,
    valid: float = 0.15,
    seed: int = 42,
    *,
    force_download: bool = False,
    remove_original: bool = True
) -> tuple[Path, Path, Path]:
    """
    """
    dest = Path(data_path) / 'mnist_test_seq.npy'

    _download_data(
        dest=dest,
        url=MOVING_MINST_URL,
        expected_sha256=MOVING_MINST_SHA256,
        force=force_download,
    ) 

    temp = np.load(dest, mmap_mode='r')
    total = temp.shape[1]

    rng = np.random.default_rng(seed)
    index = rng.permutation(total)

    n_train = int(total * train)
    n_valid = int(total * valid)

    train_index = index[:n_train]
    valid_index = index[n_train:n_train + n_valid]
    test_index = index[n_train + n_valid:]

    dest_train = Path(data_path) / 'moving_mnist_train.npy'
    dest_valid = Path(data_path) / 'moving_mnist_valid.npy'
    dest_test = Path(data_path) / 'moving_mnist_test.npy'

    np.save(dest_train, temp[:, train_index])
    np.save(dest_valid, temp[:, valid_index])
    np.save(dest_test, temp[:, test_index])

    if remove_original:
        import os
        os.remove(dest)

    return dest_train, dest_valid, dest_test


def _download_data(
    dest: Path, 
    *,
    url: str = MOVING_MINST_URL,
    expected_sha256: str | None = None,
    force: bool = False
) -> Path:
    """
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not force and _is_data_valid(dest, expected_sha256):
        logger.info('A valid cache was found in %s. It will not be downloaded again.',
            dest)
        return dest

    # try to download the data
    temp = dest.with_suffix('.part')
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info('Downloading (attempt %d/%d): %s',
                attempt, MAX_RETRIES, url)

            with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as resp:
                resp.raise_for_status()

                total_bytes = int(resp.headers.get('Content-Length', 0)) or None

                with(
                    temp.open('wb') as f,
                    tqdm(
                        total=total_bytes,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=dest.name,
                        dynamic_ncols=True,
                    ) as bar,
                ):
                    for chunk in _stream_chunks(resp):
                        f.write(chunk)
                        bar.update(len(chunk))

            break # success download
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == MAX_RETRIES:
                temp.unlink(missing_ok=True)
                raise

            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            logger.warning('Network error (%s). Trying again in %.0f s...',
                e.__class__.__name__, wait)
            time.sleep(wait)

    # check if everything is ok
    if expected_sha256 is not None:
        logger.info('Verifying file integrity...')
        actual_hash = _compute_sha256(temp)
        if actual_hash != expected_sha256:
            temp.unlink(missing_ok=True)
            raise ValueError(
                f'Invalid SHA-256 hash for "{dest.name}".\n'
                f'  Expected: {expected_sha256}\n'
                f'  Actual:   {actual_hash}'
            )

        logger.info('Integrity verified')

    # move the correct file to its destinity
    shutil.move(str(temp), dest)
    logger.info('Saved in "%s".', dest)
    return dest


def _compute_sha256(
    path: Path
) -> str:
    """
    """
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(CHUNK_SIZE), b''):
            h.update(block)
    return h.hexdigest()


def _is_data_valid(
    path: Path,
    expected_hash: str | None
) -> bool:
    """
    """
    if not path.exists():
        return False

    if expected_hash is None:
        logger.warning('The expected hash was not provided; accepting the cached file without verification')
        return True

    actual_hash = _compute_sha256(path)
    if actual_hash == expected_hash:
        return True

    logger.warning('Incorrect hash for %s\n  Expected: %s\n  Actual: %s',
        path.name, expected_hash, actual_hash)
    return False


def _stream_chunks(response: requests.Response) -> Iterator[bytes]:
    """
    """
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            yield chunk


# How to use it
# >>> from data import load_moving_mnist
# >>> x = load_moving_mnist() 
# >>> x.shape, x.dtype
# ((20, 10000, 64, 64), dtype('uint8'))
# 
# this will create a folder called "dataset"
# with in a file called mnist_test_seq.npy