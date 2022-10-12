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
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, to)

    else:
        raise ValueError("Unsupported archive extension.")

    logging.info(f"{path} extracted to {to}.")

    return to
