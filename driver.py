import hashlib
import os
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
import toml
import typer
from loguru import logger
from typer_config import use_toml_config
from typing_extensions import Annotated
from tqdm import tqdm

from GroundTruthReconstruction import ground_truth_reconstruction

app = typer.Typer()


def download_and_verify(url: str, expected_hash: str, target_dir: Path) -> Path:
    filename = target_dir / url.split("/")[-1]

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    md5_hash = hashlib.md5()
    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    md5_hash.update(chunk)
                    pbar.update(len(chunk))

    calculated_hash = md5_hash.hexdigest()
    if calculated_hash != expected_hash:
        os.remove(filename)  # Remove the file if hash doesn't match
        raise ValueError(
            f"MD5 hash mismatch. Expected: {expected_hash}, Got: {calculated_hash}"
        )

    return filename


@app.command()
@use_toml_config(default_value="config.toml")
def main(
    walnut: Annotated[int, typer.Argument(...)],
    urls: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    angular_sub_sampling: Annotated[int, typer.Option()],
    recon: Annotated[Path, typer.Option(file_okay=False)],
    angular_start_index: Annotated[int, typer.Option()] = 0,
):
    try:
        dl_info = toml.load(urls)[f"Walnut{walnut}"]
    except KeyError:
        raise ValueError(f"No URLs for Walnut{walnut} found in {urls}")

    with TemporaryDirectory() as _tmpdir:
        tmpdir = Path(_tmpdir)

        logger.info(f"Downloading and extracting Walnut{walnut} zip file from ")
        zip_path = download_and_verify(
            dl_info["url"],
            dl_info["checksum"],
            tmpdir,
        )

        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        logger.info("Running ground truth reconstructions")
        for angular_start_index in range(0, angular_sub_sampling - 1):
            ground_truth_reconstruction(
                tmpdir, walnut, angular_sub_sampling, angular_start_index, recon
            )
            logger.info(f"Ground truth reconstruction saved to {recon}")


if __name__ == "__main__":
    app()
