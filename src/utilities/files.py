import requests
import tarfile
import logging


def download(url: str, to: str) -> str:
    logging.info(f"Downloading {url} to {to}.")

    r = requests.get(url)

    with open(to, "wb") as f:
        f.write(r.content)

    logging.info(f"{url} downloaded to {to}.")

    return to


def extract(path: str, to: str) -> str:
    logging.info(f"Extracting {path} to {to}.")

    if path.endswith(".tar.gz"):
        with tarfile.open(path) as tar:
            tar.extractall(to)

    else:
        raise ValueError("Unsupported archive extension.")

    logging.info(f"{path} extracted to {to}.")

    return to
