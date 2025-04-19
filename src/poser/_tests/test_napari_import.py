# test_napari_import.py

def test_napari_basic_import():
    import napari
    assert napari.__version__ is not None
