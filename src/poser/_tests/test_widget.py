import os
from pathlib import Path

from poser import PoserWidget
import pandas as pd


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
    assert len(my_widget.chkpt_files > 0)


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
    assert len(my_widget.chkpt_files > 0)

    # load dlc file
    test_dlc_file = Path(
        os.path.join(os.getcwd(), "src/poser/_tests/test_DLC_file.h5")
    )
    my_widget.h5_picker_changed(test_dlc_file)

    assert len(viewer.layers) == 0  ## this
    assert len(my_widget.viewer1d.layers) == 0
    assert my_widget.classification_data == {}
    assert my_widget.label_menu.choices == my_widget.choices
    assert type(my_widget.dlc_data == pd.DataFrame)

    # load video


# test load video


#
