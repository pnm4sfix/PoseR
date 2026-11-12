import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from napari.components import ViewerModel
from napari.layers import Image
from poser._widget import PoserWidget
import pandas as pd
import numpy as np
import requests
import zipfile
import os
from tqdm import tqdm

def check_test_data_exists():
    #data_dir = Path(os.path.join(os.getcwd(), "data/TestData")) 
    #assert data_dir.exists(), "Test data not found. Please run `make download_test_data` to download the test data."

    

    url = "https://github.com/pnm4sfix/PoseR/releases/download/v0.0.1b4/TestData.zip"
    output_dir = "./data/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = "./data/TestData.zip"

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for failed requests

    # Save to file
    with open(output_path, "wb") as f:
        print("Downloading test data...")
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            f.write(chunk)

    print("Download complete!")

    # If it's a zip file, extract it
    

    if output_path.endswith(".zip"):
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall("data/")
        print("Extraction complete!")

    assert os.path.exists(output_path), "Test data not found. Please run `make download_test_data` to download the test data."

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    my_widget = PoserWidget(viewer)

    check_test_data_exists()

    # call our widget method
    my_widget.decoder_dir_changed(
        Path(os.path.join(os.getcwd(), "data/PoseRTestData"))
    )

    assert len(my_widget.classification_dict) > 0
    assert len(my_widget.label_menu.choices) > 0
    assert len(my_widget.ckpt_files) > 0


# test load deeplabcut data load
def test_workflow1(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    my_widget = PoserWidget(viewer)

    # load
    my_widget.decoder_dir_changed(
        Path(os.path.join(os.getcwd(), "data/PoseRTestData"))
    )

    assert len(my_widget.classification_dict) > 0
    assert len(my_widget.label_menu.choices) > 0
    assert len(my_widget.ckpt_files) > 0

    # load dlc file
    test_dlc_file = os.path.join(my_widget.decoder_data_dir, "test_DLC_file.h5" ) #Path(
        #os.path.join(os.getcwd(), "/data/PoseRTestData/test_DLC_file.h5")
    #)
    my_widget.h5_picker_changed(test_dlc_file)

    assert len(viewer.layers) == 0  ## this
    assert len(my_widget.viewer1d.layers) == 1  ## just line layer
    assert my_widget.classification_data == {}
    assert list(my_widget.label_menu.choices) == my_widget.choices
    assert type(my_widget.dlc_data) == pd.DataFrame

    # load video
    test_video_file = os.path.join(my_widget.decoder_data_dir, "test_video.avi") #Path(
        #os.path.join(os.getcwd(), "/data/PoseRTestData/test_video.avi")
    #)
    my_widget.vid_picker_changed(test_video_file)

    assert my_widget.fps == 330.0
    assert len(viewer.layers) == 1
    assert type(my_widget.im_subset) == Image

    # test change individual
    my_widget.individual_changed(1)
    assert my_widget.ind == 1
    assert len(viewer.layers) == 2
    assert len(my_widget.coords_data.keys()) > 0

    # test extract behaviours from set parameters and note speed- check viewer1d vis range
    my_widget.extract_behaviours()

    assert len(my_widget.behaviours) > 0

    # test behaviour change from 0-3 - check classification data updates, and that behaviour label is accurate
    my_widget.behaviour_changed(1)
    assert len(viewer.layers) == 3  # tracks layer will be added

    assert (
        len(my_widget.classification_data[my_widget.ind].keys()) == 0
    )  # classification data only updated when move from behaviour

    my_widget.behaviour_changed(2)
    assert len(viewer.layers) == 3  # tracks layer will be added

    assert (
        len(my_widget.classification_data[my_widget.ind].keys()) == 1
    )  # classification data only updated when move from behaviour

    # set current choice as left, change to behaviour 3, change choice then go back to
    my_widget.label_menu.value = "left"

    my_widget.behaviour_changed(3)

    assert len(my_widget.classification_data[my_widget.ind].keys()) == 2
    assert (
        my_widget.classification_data[my_widget.ind][2]["classification"]
        == "left"
    )
    my_widget.label_menu.value = "forward"

    my_widget.behaviour_changed(2)
    assert len(my_widget.classification_data[my_widget.ind].keys()) == 3
    assert (
        my_widget.classification_data[my_widget.ind][2]["classification"]
        == "left"
    )
    assert (
        my_widget.classification_data[my_widget.ind][3]["classification"]
        == "forward"
    )
    assert my_widget.label_menu.value == "left"

    # test behaviour change back to 0 -

    # test save classification_data
    before_save_data = my_widget.classification_data
    my_widget.save_to_h5(None)

    assert os.path.exists(str(my_widget.video_file) + "_classification.h5")

    # load classification_data and check its the same
    my_widget.convert_h5_todict(
        str(my_widget.video_file) + "_classification.h5"
    )
    after_save_data = my_widget.classification_data

    # delete classification.h5
    os.remove(str(my_widget.video_file) + "_classification.h5")

    assert before_save_data.keys() == after_save_data.keys()
    # break down assertions for all parts of data
    for key in before_save_data.keys():
        assert (
            before_save_data[key][1].keys() == after_save_data[key][1].keys()
        )

    # test load different file - assert all changes that need to be made


def test_predict_poses_keeps_one_property_row_per_point(monkeypatch):
    # The new keypoints were appended to the layer's points but replaced its
    # properties, so the layer had more points than property rows and napari
    # rejected it once the whole video had been processed.
    n_frames, n_nodes = 2, 3

    class FakeYOLO:
        def __init__(self, weights):
            pass

        def to(self, device):
            pass

        def track(self, **kwargs):
            return [
                SimpleNamespace(
                    keypoints=SimpleNamespace(
                        xy=torch.rand(1, n_nodes, 2), conf=torch.rand(1, n_nodes)
                    ),
                    boxes=None,
                )
                for _ in range(n_frames)
            ]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    frames = np.zeros((n_frames, 64, 64))
    # predict_poses run on a stand-in for the widget, so no Qt window is needed.
    widget = SimpleNamespace(
        initialise_params=lambda: None,
        chkpt_dropdown=SimpleNamespace(value="yolo-fake.pt"),
        accelerator="cpu",
        im=frames,
        im_subset=SimpleNamespace(data=frames),
        video_file="fake.avi",
        n_inds=1,
        points_layer=None,
        viewer=ViewerModel(),
        convert_point_layer_to_coords_data=lambda layer: None,
    )

    # The first run starts from the one-point placeholder, the second appends.
    for run in (1, 2):
        PoserWidget.predict_poses(widget)

        n_points = 1 + run * n_frames * n_nodes
        assert len(widget.points_layer.data) == n_points
        for name in ("confidence", "ind", "node"):
            assert len(widget.points_layer.properties[name]) == n_points

    np.testing.assert_array_equal(
        widget.points_layer.properties["node"][1:],
        np.tile(np.arange(n_nodes), 2 * n_frames),
    )
