"""
Pose data I/O — pure functions, no napari dependency.

Supports:
  * DeepLabCut  (.h5 / .csv)
  * SLEAP       (.h5)
  * PoseR-native (.h5 via PyTables)

All readers return a ``coords_data`` dict::

    {
        individual_key: {
            "x":  DataFrame or ndarray  (n_nodes × n_frames),
            "y":  DataFrame or ndarray  (n_nodes × n_frames),
            "ci": DataFrame or ndarray  (n_nodes × n_frames),
        },
        ...
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_dlc(
    h5_file: PathLike,
    *,
    clean: bool = False,
    confidence_threshold: float = 0.8,
    bodypoints: Optional[list[str]] = None,
) -> Dict:
    """Read a DeepLabCut .h5 or .csv pose file.

    Parameters
    ----------
    h5_file:
        Path to DLC output file (.h5 or .csv).
    clean:
        If True, NaN-interpolate low-confidence frames (ci < *confidence_threshold*).
    confidence_threshold:
        Threshold below which keypoints are masked when *clean* is True.
    bodypoints:
        Optional subset of bodypart names to keep (e.g. for OFT mouse datasets).

    Returns
    -------
    dict
        ``{individual: {"x": DataFrame, "y": DataFrame, "ci": DataFrame}}``
    """
    h5_file = str(h5_file)
    if h5_file.endswith(".h5"):
        dlc_data = pd.read_hdf(h5_file)
        data_t = dlc_data.transpose()
        try:
            data_t["individuals"]
            data_t = data_t.reset_index()
        except KeyError:
            data_t["individuals"] = ["individual1"] * data_t.shape[0]
            data_t = (
                data_t.reset_index()
                .set_index(["scorer", "individuals", "bodyparts", "coords"])
                .reset_index()
            )
    elif h5_file.endswith(".csv"):
        dlc_data = pd.read_csv(h5_file, header=[0, 1, 2], index_col=0)
        data_t = dlc_data.transpose()
        data_t["individuals"] = ["individual1"] * data_t.shape[0]
        data_t = (
            data_t.reset_index()
            .set_index(["scorer", "individuals", "bodyparts", "coords"])
            .reset_index()
        )
    else:
        raise ValueError(f"Unsupported DLC file format: {h5_file}")

    coords_data: Dict = {}
    for individual in data_t.individuals.unique():
        if bodypoints is not None:
            indv = data_t[
                (data_t.individuals == individual)
                & (data_t.bodyparts.isin(bodypoints))
            ]
        else:
            indv = data_t[data_t.individuals == individual].copy()

        if clean:
            indv.loc[:, 0:] = indv.loc[:, 0:].interpolate(axis=1)

        x = indv.loc[indv.coords == "x", 0:].reset_index(drop=True)
        y = indv.loc[indv.coords == "y", 0:].reset_index(drop=True)
        ci = indv.loc[indv.coords == "likelihood", 0:].reset_index(drop=True)

        if clean:
            x[ci < confidence_threshold] = np.nan
            y[ci < confidence_threshold] = np.nan
            x = x.interpolate(axis=1)
            y = y.interpolate(axis=1)

        coords_data[individual] = {"x": x, "y": y, "ci": ci}

    return coords_data


def read_sleap(h5_file: PathLike) -> Dict:
    """Read a SLEAP pose-estimation .h5 file.

    Expects groups keyed by individual, each containing datasets ``"x"``,
    ``"y"``, ``"ci"``.
    """
    import h5py

    coords_data: Dict = {}
    with h5py.File(str(h5_file), "r") as f:
        for individual in f.keys():
            g = f[individual]
            coords_data[individual] = {
                "x": g["x"][:],
                "y": g["y"][:],
                "ci": g["ci"][:],
            }
    return coords_data


def read_poser_coords(h5_file: PathLike) -> Dict:
    """Read a PoseR-native coords .h5 file (PyTables).

    Expects groups keyed by individual with an array named ``"coords"``
    of shape ``(3, n_nodes, n_frames)`` corresponding to [x, y, ci].
    """
    import tables as tb

    coords_data: Dict = {}
    with tb.open_file(str(h5_file), mode="r") as f:
        for ind in f.list_nodes("/"):
            arr = ind.coords[:]
            coords_data[ind._v_name] = {
                "x": arr[0],
                "y": arr[1],
                "ci": arr[2],
            }
    return coords_data


def read_coords(
    h5_file: PathLike,
    *,
    clean: bool = False,
    confidence_threshold: float = 0.8,
    bodypoints: Optional[list[str]] = None,
) -> Dict:
    """Auto-detect format and read a pose file.

    Tries DLC → SLEAP → PoseR-native in order.

    Returns
    -------
    dict
        coords_data compatible with the rest of the pipeline.
    """
    h5_file_str = str(h5_file)

    if "poser_coords" in h5_file_str:
        return read_poser_coords(h5_file)

    try:
        return read_dlc(
            h5_file,
            clean=clean,
            confidence_threshold=confidence_threshold,
            bodypoints=bodypoints,
        )
    except Exception:
        pass

    try:
        return read_sleap(h5_file)
    except Exception:
        pass

    return read_poser_coords(h5_file)


# ---------------------------------------------------------------------------
# Classification h5 (PyTables)
# ---------------------------------------------------------------------------

def read_classification_h5(filepath: PathLike) -> Dict:
    """Read a PoseR classification .h5 file.

    Returns
    -------
    dict
        ``{ind_int: {behaviour_int: {"classification": str, "coords": ndarray,
                                      "ci": ndarray, "start": int, "stop": int}}}``
    """
    import tables as tb

    classification_data: Dict = {}
    with tb.open_file(str(filepath), mode="r") as f:
        for group_name in f.root.__getattr__("_v_groups"):
            ind = f.root[group_name]
            behaviour_dict: Dict = {}
            arrays = {int(a.name): a for a in f.list_nodes(ind, classname="Array")}

            table = f.list_nodes(ind, classname="Table")[0]

            behaviours, classifications, starts, stops = [], [], [], []
            for row in table.iterrows():
                behaviours.append(row["number"])
                classifications.append(row["classification"])
                starts.append(row["start"])
                stops.append(row["stop"])

            for bhv, cls, start, stop in zip(behaviours, classifications, starts, stops):
                behaviour_dict[bhv + 1] = {
                    "classification": cls.decode("utf-8") if isinstance(cls, bytes) else cls,
                    "coords": arrays[bhv][:, :3],
                    "start": int(start),
                    "stop": int(stop),
                    "ci": arrays[bhv][:, 3],
                }

            classification_data[int(group_name)] = behaviour_dict

    return classification_data


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def save_to_h5(
    classification_data: Dict,
    video_file: PathLike,
    n_nodes: int,
    behaviour_schema,  # PyTables IsDescription subclass
) -> str:
    """Write classification_data to a PoseR .h5 file.

    Parameters
    ----------
    classification_data:
        ``{ind: {behaviour: {"classification": str, "coords": ndarray,
                              "ci": ndarray, "start": int, "stop": int}}}``
    video_file:
        Used to derive the output filename (``<video_file>_classification.h5``).
    n_nodes:
        Number of skeleton nodes.
    behaviour_schema:
        PyTables ``IsDescription`` subclass defining the label table schema.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    import tables as tb

    filename = str(video_file) + "_classification.h5"
    mode = "a" if os.path.exists(filename) else "w"
    with tb.open_file(filename, mode=mode, title="classification") as f:
        for ind, ind_subset in classification_data.items():
            ind_group = f.create_group("/", str(ind), f"Individual{ind}")
            ind_table = f.create_table(
                ind_group, "labels", behaviour_schema, f"Individual {ind} Behaviours"
            )

            for behaviour, data in ind_subset.items():
                try:
                    arr = data["coords"]
                    ci = data["ci"]
                    array = np.concatenate((arr, ci.reshape(-1, ci.shape[0]).T), axis=1)
                    f.create_array(ind_group, str(behaviour), array, f"Behaviour{behaviour}")
                    row = ind_table.row
                    row["number"] = behaviour
                    row["classification"] = data["classification"]
                    row["n_nodes"] = n_nodes
                    row["start"] = data["start"]
                    row["stop"] = data["stop"]
                    row.append()
                    ind_table.flush()
                except Exception as exc:
                    print(f"  Warning: could not save behaviour {behaviour}: {exc}")

    return filename


def save_coords_to_h5(coords_data: Dict, video_file: PathLike) -> str:
    """Write coords_data to a PoseR-native coords .h5 file.

    Parameters
    ----------
    coords_data:
        ``{individual: {"x": array, "y": array, "ci": array}}``.
    video_file:
        Used to derive output filename (``<video_file>_poser_coords.h5``).

    Returns
    -------
    str
        Absolute path to the written file.
    """
    import tables as tb

    filename = str(video_file) + "_poser_coords.h5"
    mode = "a" if os.path.exists(filename) else "w"
    with tb.open_file(filename, mode=mode, title="coords") as f:
        for ind, data in coords_data.items():
            ind_group = f.create_group("/", str(ind), f"Individual{ind}")
            coords_array = np.array([data["x"], data["y"], data["ci"]])
            f.create_array(ind_group, "coords", coords_array, "Coordinates")

    return filename


# ---------------------------------------------------------------------------
# Numpy convenience
# ---------------------------------------------------------------------------

def convert_dlc_to_ctvm(dlc_file: PathLike) -> np.ndarray:
    """Convert a DLC .h5/.csv file directly to a ``(C, T, V, M)`` numpy array.

    C=3 (x, y, ci), T=frames, V=bodyparts, M=individuals.
    """
    dlc_file = str(dlc_file)
    if dlc_file.endswith(".h5"):
        dlc_data = pd.read_hdf(dlc_file)
    elif dlc_file.endswith(".csv"):
        dlc_data = pd.read_csv(dlc_file, header=[0, 1, 2], index_col=0)
    else:
        raise ValueError(f"Unsupported file format: {dlc_file}")

    data_t = dlc_data.transpose()
    data_t["individuals"] = ["individual1"] * data_t.shape[0]
    data_t = (
        data_t.reset_index()
        .set_index(["scorer", "individuals", "bodyparts", "coords"])
        .reset_index()
    )

    ctvms = []
    bodyparts = data_t.bodyparts.unique()

    x = data_t[data_t.coords == "x"].loc[:, 0:].to_numpy()
    x = x.reshape(1, len(bodyparts), -1, 1)

    y = data_t[data_t.coords == "y"].loc[:, 0:].to_numpy()
    y = y.reshape(1, len(bodyparts), -1, 1)

    ci = data_t[data_t.coords == "likelihood"].loc[:, 0:].to_numpy()
    ci = ci.reshape(1, len(bodyparts), -1, 1)

    CVTM = np.concatenate([x, y, ci], axis=0)
    CTVM = np.swapaxes(CVTM, 1, 2)
    return CTVM
