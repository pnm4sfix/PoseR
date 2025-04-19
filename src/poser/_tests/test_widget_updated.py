import os
from pathlib import Path
from napari.layers import Image
from poser import PoserWidget
import pandas as pd
import numpy as np
import requests
import zipfile
import os
from tqdm import tqdm
import pytest
from unittest.mock import MagicMock, patch

# Code below tests a number of aspects from the funtion PLOT BEHAVIOUR TIMELINE 
# If an identified behaviour is present in the timeline
# If behaviour exists, add_region is called and is expected to have parameters 
# The mouse connection is correctly assigned

class MockBehaviorPlotter:
    def __init__(self, behaviours=None):
        self.behaviours = behaviours or []
        self.viewer1d = MagicMock()
        self.viewer1d.layers = MagicMock()
        self.viewer1d.add_region = MagicMock()
        self.viewer1d.canvas.events.mouse_press = MagicMock()
        self.regions_layer = None

    def plot_behavior_timeline(self):
        if not self.behaviours:
            print("No behaviors detected.")
            return

        if hasattr(self, "regions_layer") and self.regions_layer:
            self.viewer1d.layers.remove(self.regions_layer)

        self.regions = [([start, stop], "vertical") for start, stop in self.behaviours]
        self.regions_layer = self.viewer1d.add_region(
            self.regions, color="green", opacity=0.4, name="Behavior Timeline"
        )
        self.viewer1d.canvas.events.mouse_press.connect = MagicMock()

@pytest.fixture
def mock_plotter():
    return MockBehaviorPlotter()

def test_no_behaviors(mock_plotter, capsys):
    mock_plotter.plot_behavior_timeline()
    captured = capsys.readouterr()
    assert "No behaviors detected." in captured.out

def test_plot_behavior(mock_plotter):
    mock_plotter.behaviours = [(0, 5), (10, 15)]
    mock_plotter.plot_behavior_timeline()
    
    # Ensure add_region is called with correct data
    expected_regions = [([0, 5], "vertical"), ([10, 15], "vertical")]
    mock_plotter.viewer1d.add_region.assert_called_once_with(
        expected_regions, color="green", opacity=0.4, name="Behavior Timeline"
    )
    
    # Ensure mouse event connection is set
    assert mock_plotter.viewer1d.canvas.events.mouse_press.connect.called



# Code below tests a number of aspects from the funtion: JUMP TO EVENT && ADD BEHAVIOUR FROM SELECTED AREA 
# simulates unittest.mock for exp behaviours, verify function calls and ensure expected results 

class TestJumpToEvent:
    
    @pytest.fixture
    def mock_instance(self):
        class MockClass:
            def __init__(self):
                self.viewer = MagicMock()
                self.viewer.dims = MagicMock()
                self.behaviours = [(5,15), (25, 35)] # example behaviours we've jumped to?? 
                self.add_behaviour_from_selected_area = MagicMock()
        
            def jump_to_event(self, event):
                if event.inaaxes and self.behaviours: 
                    x_click = event.xdata
                    closest_idx = np.argmin(np.abs([start for start, _ in self.behaviours] - x_click))
                    frame = int(self.behaviours[closest_idx][0])
                    self.viewer.dims.set_point(0, frame)
                    start, stop = self.behaviours[closest_idx]
                    if not (start <= x_click <= stop):
                        self.add_behaviour_from_selected_area()
                    
        return MockClass()

    def test_jump_to_existing_behaviour(self, mock_instance):
        event = MagicMock(inaxes = True, xdata=10)
        mock_instance.jump_to_event(event)
        mock_instance.viewer.dims_set_point.asset_called()
        mock_instance.add_behaviour_from_selected_area.assert_not_called()
    
    def test_jump_to_new_behaviour(self, mock_instance): 
        event = MagicMock(inaxes=True, xdata=10)
        mock_instance.jump_to_event(event)
        mock_instance.add_behaviour_from_selected_area.assert_called_once()
        




# Code below tests a number of aspects from the funtion: Re-TRAIN 
# correct training pipeline, checking the correct associated outputs for each step

class TestTrainFunction: 
    
    @pytest.fixture
    def mock_instance(self):
        class MockClass:
            def __init__(self):
                    self.decoder_data_dir = "test_dir"
                    self.initalise_params = MagicMock(return_valie=({}, {}, {}))
                    self.load_corrected_classification_data = MagicMock(return_value={})
                    self.populate_chkpt_dropdown = MagicMock()
                    self.finetune_model = MagicMock()
                    self.prepare_data = MagicMock()
                
            def train(self, use_corrections=False):
                if not os.path.exists(os.path.join(self.decoder_data_dir, "Zebtrain.npy")):
                    self.prepare_data()
                data_cfg, _, _ = self.initalise_params()
                if use_corrections:
                    data_cfg["corrected_labels"] = self.load_corrected_classification_data()
                self.populate_chkpt_dropdown()
                self.finetune_model()
                
        return MockClass()
        
    def test_train_without_corrections(self, mock_instance):
        with patch("os.path.exists", return_value=True):
            mock_instance.train(use_corrections=False)
            mock_instance.initialise_params.assert_called_once()
            mock_instance.populate_chkpt_dropdown.assert_called_once()
            mock_instance.finetune_model.assert_called_once()
            mock_instance.load_corrected_classification_data.assert_not_called()

    def test_train_with_corrections(self, mock_instance):
        with patch("os.path.exists", return_value=True):
            mock_instance.train(use_corrections=True)
            mock_instance.initialise_params.assert_called_once()
            mock_instance.load_corrected_classification_data.assert_called_once()
            mock_instance.populate_chkpt_dropdown.assert_called_once()
            mock_instance.finetune_model.assert_called_once()

    def test_train_prepares_data_when_missing(self, mock_instance):
        with patch("os.path.exists", return_value=False):
            mock_instance.train()
            mock_instance.prepare_data.assert_called_once()






