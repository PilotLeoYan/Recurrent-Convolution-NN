from pathlib import Path
import requests
from tqdm import tqdm
import time
from typing import Iterator
import shutil

try:
    from utils.logger import get_logger
except ModuleNotFoundError:
    from ..utils.logger import get_logger

logger = get_logger(__name__)


CHUNK_SIZE: int = 8 * 1024 * 1024 # 8 MiB
MAX_RETRIES: int = 3
RETRY_BACKOFF: float = 2.0 # seconds
REQUEST_TIMEOUT: tuple[int, int] = (10, 60) # (connect, read) seconds
MOVING_MINST_URL: str = ('https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy')


def download_moving_mnist(
    dir_path: str = r'dataset',
    train: float = 0.7,
    valid: float = 0.15,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """
    """
    assert train + valid < 1, f'train + valid porcentage must be less than 1. Receive train: {train}, valid: {valid}.'
    
    dest = Path(dir_path)
    
    train_path = dest / 'moving_mnist_train.npy'
    valid_path = dest / 'moving_mnist_valid.npy'
    test_path = dest / 'moving_mnist_test.npy'

    download_data: bool = False

    if not train_path.is_file():
        download_data = True
    elif not valid_path.is_file():
        download_data = True
    elif not test_path.is_file():
        download_data = True

    if not download_data:
        logger.info('Dataset already exists. Omitting download.')
        return train_path, valid_path, test_path
    
    original_path = dest / 'mnist_test_seq.npy'
    original_path.parent.mkdir(parents=True, exist_ok=True)

    _download_original(original_path, MOVING_MINST_URL)
    
    _split_dataset(
        original_path,
        train_path,
        valid_path,
        test_path,
        train, valid, seed
    )

    if original_path.is_file():
        import os
        os.remove(original_path)
        logger.info('Remove original dataset "%s"', original_path)

    return train_path, valid_path, test_path


def _split_dataset(
    original_path: Path,
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    train: float,
    valid: float,
    seed: int = 42,
):
    """
    """
    logger.info('Splitting dataset...')
    
    import numpy as np

    temp = np.load(original_path, mmap_mode='r')
    total = temp.shape[1]

    rng = np.random.default_rng(seed)
    index = rng.permutation(total)

    n_train = int(total * train)
    n_valid = int(total * valid)

    train_index = index[:n_train]
    valid_index = index[n_train:n_train + n_valid]
    test_index = index[n_train + n_valid:]

    logger.info('Saving splittings...')
    np.save(train_path, temp[:, train_index])
    np.save(valid_path, temp[:, valid_index])
    np.save(test_path, temp[:, test_index])


def _download_original(
    dest: Path,
    url: str,
) -> None:
    """
    """
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

    shutil.move(str(temp), dest)
    

def _stream_chunks(response: requests.Response) -> Iterator[bytes]:
    """
    """
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            yield chunk