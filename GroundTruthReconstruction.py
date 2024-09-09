#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jul 25 15:38:08 2018

@author: Henri Der Sarkissian, Felix Lucka, CWI, Amsterdam
henri.dersarkissian@gmail.com
Felix.Lucka@cwi.nl

This python script computes an iterative reconstruction for the full
data of one of the data sets described in
"A Cone-Beam X-Ray CT Data Collection Designed for Machine Learning" by
Henri Der Sarkissian, Felix Lucka, Maureen van Eijnatten,
Giulia Colacicco, Sophia Bethany Coban, Kees Joost Batenburg

MODIFIED by @smutch 2024-09-09
- Fixed numpy v2 compatibility issues
- Added angular_start_index parameter and labeled outputs appropriately
- Wrapped code in a function for reusability
- Moved paths to pathlib.Path objects
- Fixed general formatting and linting issues
"""

import time
from pathlib import Path

import astra
import imageio
import matplotlib.pyplot as plt
import numpy as np

import NesterovGradient

# select the orbits you want to reconstruct the data from:
# 1 higher source position, 2 middle source position, 3 lower source position
ORBITS_TO_RECON = [1, 2, 3]
# select of voxels per mm in one direction (higher = larger res)
# (all reference reconstructions are computed with 10)
VOXEL_PER_MM = 10


def ground_truth_reconstruction(
    data_path: Path,
    walnut_id: int,
    angular_sub_sampling: int,
    angular_start_index: int,
    recon_path: Path | None = None,
):
    if recon_path is None:
        recon_path = data_path / "reconstructions"

    #### load and pre-process data #################################################

    t = time.time()
    print("load and pre-process data", flush=True)

    # we add the info about walnut
    data_path_full = data_path / f"Walnut{walnut_id}/Projections"

    projs_name = "scan_{:06}.tif"
    dark_name = "di000000.tif"
    flat_name = ["io000000.tif", "io000001.tif"]
    vecs_name = "scan_geom_corrected.geom"
    projs_rows = 972
    projs_cols = 768

    # Create the numpy array which will receive projection data from tiff files
    projs = np.zeros((projs_rows, 0, projs_cols), dtype=np.float32)

    # And create the numpy array receiving the motor positions read from the geometry file
    vecs = np.zeros((0, 12), dtype=np.float32)
    nb_projs_orbit = len(range(angular_start_index, 1200, angular_sub_sampling))
    # projection file indices, we need to read in the projection in reverse order due to the portrait mode acquision
    projs_idx = range(1200, angular_start_index, -angular_sub_sampling)

    # transformation to apply to each image, we need to get the image from
    # the way the scanner reads it out into to way described in the projection
    # geometry
    def trafo(image):
        return np.transpose(np.flipud(image))

    # Loop over the subset of orbits we want to load at the same time
    for orbit_id in ORBITS_TO_RECON:
        orbit_data_path = data_path_full / f"tubeV{orbit_id}"

        # load the numpy array describing the scan geometry of the orbit from file
        vecs_orbit = np.loadtxt(orbit_data_path / vecs_name)

        # get the positions we need; there are in fact 1201, but the last and first one come from the same angle
        vecs = np.concatenate(
            (vecs, vecs_orbit[range(0, 1200, angular_sub_sampling)]), axis=0
        )

        # load flat-field and dark-fields
        # there are two flat-field images (taken before and after acquisition), we simply average them
        dark = trafo(imageio.imread(orbit_data_path / dark_name))
        flat = np.zeros((2, projs_rows, projs_cols), dtype=np.float32)
        for i, fn in enumerate(flat_name):
            flat[i] = trafo(imageio.imread(orbit_data_path / fn))
        flat = np.mean(flat, axis=0)

        # load projection data directly on the big projection array
        projs_orbit = np.zeros(
            (nb_projs_orbit, projs_rows, projs_cols), dtype=np.float32
        )
        for i in range(nb_projs_orbit):
            projs_orbit[i] = trafo(
                imageio.imread(orbit_data_path / projs_name.format(projs_idx[i]))
            )

        # subtract the dark field, devide by the flat field, and take the negative log to linearize the data according to the Beer-Lambert law
        projs_orbit -= dark
        projs_orbit /= flat - dark

        # take negative log
        np.log(projs_orbit, out=projs_orbit)
        np.negative(projs_orbit, out=projs_orbit)

        # permute data to ASTRA convention
        projs_orbit = np.transpose(projs_orbit, (1, 0, 2))

        # attach to projs
        projs = np.concatenate((projs, projs_orbit), axis=1)
        del projs_orbit

    projs = np.ascontiguousarray(projs)

    print(np.round(time.time() - t, 3), "sec elapsed")

    ### compute iterative reconstruction ###########################################

    t = time.time()
    print("compute reconstruction:", flush=True)

    # size of the reconstruction volume in voxels
    vol_sz = 3 * (50 * VOXEL_PER_MM + 1,)
    # size of a cubic voxel in mm
    vox_sz = 1 / VOXEL_PER_MM
    # numpy array holding the reconstruction volume
    vol_rec = np.zeros(vol_sz, dtype=np.float32)

    # we need to specify the details of the reconstruction space to ASTRA
    # this is done by a "volume geometry" type of structure, in the form of a Python dictionary
    # by default, ASTRA assumes a voxel size of 1, we need to scale the reconstruction space here by the actual voxel size
    vol_geom = astra.create_vol_geom(vol_sz)
    vol_geom["option"]["WindowMinX"] = vol_geom["option"]["WindowMinX"] * vox_sz
    vol_geom["option"]["WindowMaxX"] = vol_geom["option"]["WindowMaxX"] * vox_sz
    vol_geom["option"]["WindowMinY"] = vol_geom["option"]["WindowMinY"] * vox_sz
    vol_geom["option"]["WindowMaxY"] = vol_geom["option"]["WindowMaxY"] * vox_sz
    vol_geom["option"]["WindowMinZ"] = vol_geom["option"]["WindowMinZ"] * vox_sz
    vol_geom["option"]["WindowMaxZ"] = vol_geom["option"]["WindowMaxZ"] * vox_sz

    # we need to specify the details of the projection space to ASTRA
    # this is done by a "projection geometry" type of structure, in the form of a Python dictionary
    proj_geom = astra.create_proj_geom("cone_vec", projs_rows, projs_cols, vecs)

    # register both volume and projection geometries and arrays to ASTRA
    vol_id = astra.data3d.link("-vol", vol_geom, vol_rec)
    proj_id = astra.data3d.link("-sino", proj_geom, projs)

    projector_id = astra.create_projector("cuda3d", proj_geom, vol_geom)

    ## finally, create an ASTRA configuration.
    ## this configuration dictionary setups an algorithm, a projection and a volume
    ## geometry and returns a ASTRA algorithm, which can be run on its own
    astra.plugin.register(NesterovGradient.AcceleratedGradientPlugin)
    cfg_agd = astra.astra_dict("AGD-PLUGIN")
    cfg_agd["ProjectionDataId"] = proj_id
    cfg_agd["ReconstructionDataId"] = vol_id
    cfg_agd["ProjectorId"] = projector_id
    cfg_agd["option"] = {}  # pyright: ignore[reportArgumentType]
    cfg_agd["option"]["MinConstraint"] = 0
    alg_id = astra.algorithm.create(cfg_agd)

    # Run Nesterov Accelerated Gradient Descent algorithm with 'nb_iter' iterations
    nb_iter = 50
    astra.algorithm.run(alg_id, nb_iter)

    # release memory allocated by ASTRA structures
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(vol_id)

    print(np.round(time.time() - t, 3), "sec elapsed")

    ### plot and save reconstruction ##############################################

    t = time.time()
    print("save results", flush=True)

    # low level plotting
    f, ax = plt.subplots(1, 3, sharex=False, sharey=False)
    ax[0].imshow(vol_rec[vol_sz[0] // 2, :, :])
    ax[1].imshow(vol_rec[:, vol_sz[1] // 2, :])
    ax[2].imshow(vol_rec[:, :, vol_sz[2] // 2])
    f.tight_layout()

    # construct full path for storing the results
    recon_path_full = (
        recon_path
        / f"Walnut{walnut_id}/ass{angular_sub_sampling}_asi{angular_start_index}"
    )

    # create the directory in case it doesn't exist yet
    if not recon_path_full.exists():
        recon_path_full.mkdir(parents=True)

    # Save every slice in  the volume as a separate tiff file
    orbit_str = "pos"
    for orbit_id in ORBITS_TO_RECON:
        orbit_str = orbit_str + "{}".format(orbit_id)

    for i in range(vol_sz[0]):
        slice_path = (
            recon_path_full
            / f"nnls_{orbit_str}_iter{nb_iter}_ass{angular_sub_sampling}_asi{angular_start_index}_vmm{VOXEL_PER_MM}_{i:06d}.tiff"
        )
        imageio.imwrite(slice_path, vol_rec[i, ...])

    print(np.round(time.time() - t, 3), "sec elapsed")
