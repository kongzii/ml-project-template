import mlflow
import argparse

from utilities.tracking import init_mlflow
from utilities.files import download, extract

COMPONENT_NAME = "dataset"


def main(args: argparse.Namespace):
    dataset_archive = download(
        args.dataset_url,
        args.datas_dir + "/dataset.tar.gz"
    )
    extract(dataset_archive, args.datas_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datas-dir", default="/datas")
    parser.add_argument("--dataset-url", default="https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz")

    args = parser.parse_args()

    with init_mlflow(run_name=COMPONENT_NAME, args=args):
        main(args)
