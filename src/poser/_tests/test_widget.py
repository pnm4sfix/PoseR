import os
from pathlib import Path
from napari.layers import Image
from poser import PoserWidget
import pandas as pd
import numpy as np


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    my_widget = PoserWidget(viewer)

    # call our widget method
    my_widget.decoder_dir_changed(
        Path(os.path.join(os.getcwd(), "src/poser/_tests"))
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
        Path(os.path.join(os.getcwd(), "src/poser/_tests"))
    )

    assert len(my_widget.classification_dict) > 0
    assert len(my_widget.label_menu.choices) > 0
    assert len(my_widget.ckpt_files) > 0

    # load dlc file
    test_dlc_file = Path(
        os.path.join(os.getcwd(), "src/poser/_tests/test_DLC_file.h5")
    )
    my_widget.h5_picker_changed(test_dlc_file)

    assert len(viewer.layers) == 0  ## this
    assert len(my_widget.viewer1d.layers) == 1  ## just line layer
    assert my_widget.classification_data == {}
    assert list(my_widget.label_menu.choices) == my_widget.choices
    assert type(my_widget.dlc_data) == pd.DataFrame

    # load video
    test_video_file = Path(
        os.path.join(os.getcwd(), "src/poser/_tests/test_video.avi")
    )
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
