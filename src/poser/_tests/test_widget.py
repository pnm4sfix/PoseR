import os
from pathlib import Path

from poser import PoserWidget


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
