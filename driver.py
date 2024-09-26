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

import numpy as np
import xarray as xr
import cv2
from PIL import Image

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


def index_from_path(path):
    return int(path.stem.split("_")[-1])


def read_slice(path):
    with Image.open(path) as fd:
        im = xr.DataArray(
            (_vals := np.array(fd)),
            coords=[
                ("j", (_ind := np.indices(_vals.shape, sparse=True))[0].ravel()),
                ("i", _ind[1].ravel()),
            ],
            dims=["j", "i"],
        )
    return im


def crop_recons(input_path_full: Path, pad: int = 1) -> xr.DataArray:
    paths = sorted(input_path_full.iterdir(), key=index_from_path)
    cube = xr.concat([read_slice(p) for p in paths], dim="k")
    cube = cube.assign_coords({"k": np.arange(cube.shape[0])})
    cube.name = "intensity"

    thresh_frac = (cube > 0.06).sum(("i", "j")) / cube.i.size / cube.j.size
    valid_k = cube.k[thresh_frac > 0.006]

    bbox = [cube.i.size, cube.j.size, 0, 0]
    for k in tqdm(valid_k):
        _ret, thresh = cv2.threshold(
            cv2.GaussianBlur(
                (cube.sel(k=k).values / cube.max().item() * 255.0).astype(np.uint8),
                (5, 5),
                0,
            ),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        thresh = cv2.erode(thresh, None)
        x, y, w, h = cv2.boundingRect(thresh)
        bbox[0] = min(bbox[0], x)
        bbox[1] = min(bbox[1], y)
        bbox[2] = max(bbox[2], x + w)
        bbox[3] = max(bbox[3], y + h)

    return cube.sel(
        k=valid_k,
        i=np.s_[bbox[0] - pad : bbox[2] + pad],
        j=np.s_[bbox[1] - pad : bbox[3] + pad],
    )


@app.command()
@use_toml_config(default_value="config.toml")
def main(
    walnut: Annotated[int, typer.Argument(...)],
    urls: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    angular_sub_sampling: Annotated[int, typer.Option()],
    recon: Annotated[Path, typer.Option(file_okay=False)],
    input_dir: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    angular_start_index: Annotated[int | None, typer.Option()] = None,
):
    tmpdir: TemporaryDirectory[str] | None = None

    if input_dir is None:
        try:
            dl_info = toml.load(urls)[f"Walnut{walnut}"]
        except KeyError:
            raise ValueError(f"No URLs for Walnut{walnut} found in {urls}")

        tmpdir = TemporaryDirectory()
        input_dir = Path(tmpdir.name)

        logger.info(
            f"Downloading and extracting Walnut{walnut} zip file from {dl_info['url']}"
        )
        zip_path = download_and_verify(
            dl_info["url"],
            dl_info["checksum"],
            input_dir,
        )

        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(input_dir)


    work_dir = input_dir / "recon_workdir"
    logger.info("Running ground truth reconstructions")

    def process(ass: int, asi: int) -> None:
        recon_path_full = ground_truth_reconstruction(
            input_dir, walnut, ass, asi, work_dir
        )
        out_dir = recon_path_full
        cube = crop_recons(recon_path_full)
        out_dir = recon.joinpath(*recon_path_full.parts[-2:])
        out_dir.mkdir(parents=True, exist_ok=True)
        cube.to_netcdf(out_dir / f"ass{ass}_asi{asi}.nc")
        logger.info(f"walnut={walnut}, ass={ass}, asi={asi} complete: {out_dir}")

    for asi in (
        range(0, angular_sub_sampling)
        if angular_start_index is None
        else [angular_start_index]
    ):
        process(angular_sub_sampling, asi)

    if angular_start_index is None:
        process(1, 0)

    if tmpdir is not None:
        tmpdir.cleanup()


if __name__ == "__main__":
    app()
