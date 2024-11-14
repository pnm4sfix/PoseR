"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""

from ast import Try
from cProfile import label
from re import I, L
import sys
from tkinter import W

sys.path.insert(1, "./")
import os

# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max
# from scipy import ndimage as ndi
# from sklearn.manifold import TSNE
# from matplotlib.animation import FuncAnimation
import time
from typing import TYPE_CHECKING

import napari_plot
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as st
import tables as tb
import torch
import torch.nn as nn
import yaml
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    FloatSpinBox,
    Label,
    PushButton,
    SpinBox,
    TextEdit,
)
from napari_plot._qt.qt_viewer import QtViewer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.stochastic_weight_avg import (
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.ndimage import gaussian_filter1d as gaussian_filter1d
from scipy.signal import find_peaks
from torch.utils.data import DataLoader

from ._loader import HyperParams, ZebData
from .models import c3d, st_gcn_aaai18_pylightning_3block


try:
    import pygmtools as pygm
except:
    print("pygmtools not installed")
import networkx as nx

try:
    from napari_video.napari_video import VideoReaderNP
except:
    print("no module named napari_video. pip install napari_video")
    import time

try:
    pass
except:
    print("no cuda support")


if TYPE_CHECKING:
    pass


class PoserWidget(Container):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Add behaviour labels to a list

        self.title_label = Label(label="Settings")

        self.decoder_dir_picker = FileEdit(
            label="Select data folder",
            value="./",
            tooltip="Select data folder",
            mode="d",
        )
        self.decoder_dir_picker.changed.connect(self.decoder_dir_changed)

        self.add_behaviour_text = TextEdit(label="Enter new behavioural label")
        self.add_behaviour_button = PushButton(label="Add new behaviour label")
        self.add_behaviour_button.clicked.connect(self.add_behaviour)

        self.label_menu = ComboBox(
            label="Behaviour labels",
            choices=[],
            tooltip="Select behaviour label",
        )
        push_button = PushButton(label="Save Labels")
        self.extend(
            [
                self.decoder_dir_picker,
                self.add_behaviour_text,
                self.add_behaviour_button,
                self.label_menu,
                push_button,
            ]
        )  # self.title_label,

        # Number of nodes in pose estimation
        self.n_node_select = SpinBox(label="Number of nodes")
        self.n_node_select.changed.connect(self.set_n_nodes)
        # Center node
        self.center_node_select = SpinBox(label="Center node")
        self.center_node_select.changed.connect(self.set_center_node)

        # file pickers
        label_txt_picker = FileEdit(
            label="Select a labeled txt file",
            value="./",
            tooltip="Select labeled txt file",
        )
        self.label_h5_picker = FileEdit(
            label="Select a labeled h5 file",
            value="./",
            tooltip="Select labeled h5 file",
        )
        self.h5_picker = FileEdit(
            label="Select a DLC h5 file", value="./", tooltip="Select h5 file"
        )
        self.vid_picker = FileEdit(
            label="Select the corresponding raw video",
            value="./",
            tooltip="Select corresponding raw video",
        )
        self.extend(
            [
                self.h5_picker,
                self.vid_picker,
                self.label_h5_picker,
                label_txt_picker,
            ]
        )

        # Behavioural extraction method
        self.behavioural_extract_method = ComboBox(
            label="Behaviour extraction method",
            choices=["orth", "egocentric"],
            tooltip="Select preferred method for extracting behaviours (orth is best for zebrafish)",
        )

        self.extract_method = self.behavioural_extract_method.value
        self.behavioural_extract_method.changed.connect(
            self.update_extract_method
        )
        self.extract_behaviour_button = PushButton(
            label="Extract behaviour bouts"
        )
        self.extract_behaviour_button.clicked.connect(self.extract_behaviours)
        self.add_behaviour_from_selected_area_button = PushButton(
            label="Add behaviour from selected area"
        )
        self.add_behaviour_from_selected_area_button.clicked.connect(
            self.add_behaviour_from_selected_area
        )
        self.confidence_threshold_spinbox = FloatSpinBox(
            label="Confidence Threshold",
            tooltip="Change confidence threshold for pose estimation",
        )
        self.amd_threshold_spinbox = FloatSpinBox(
            label="Movement Threshold",
            tooltip="Change movement threshold for pose estimation",
        )

        self.amd_threshold = 2
        self.confidence_threshold = 0.8
        self.confidence_threshold_spinbox.value = self.confidence_threshold
        self.amd_threshold_spinbox.value = self.amd_threshold

        self.confidence_threshold_spinbox.changed.connect(
            self.extract_behaviours
        )
        self.amd_threshold_spinbox.changed.connect(self.extract_behaviours)

        ### Variables to define
        self.classification_data = {}
        self.point_subset = np.array([])

        self.coords_data = {}
        self.spinbox = SpinBox(
            label="Behaviour Number", tooltip="Change behaviour"
        )
        self.ind_spinbox = SpinBox(
            label="Individual Number", tooltip="Change individual"
        )

        self.extend(
            [
                self.confidence_threshold_spinbox,  # self.n_node_select, self.center_node_select,
                self.amd_threshold_spinbox,
                self.behavioural_extract_method,
                self.ind_spinbox,
                self.extract_behaviour_button,
                self.add_behaviour_from_selected_area_button,
                self.spinbox,
            ]
        )

        self.labeled_txt = label_txt_picker.value
        self.labeled_h5 = self.label_h5_picker.value
        self.h5_file = self.h5_picker.value
        self.video_file = self.vid_picker.value

        self.h5_picker.changed.connect(self.h5_picker_changed)
        self.vid_picker.changed.connect(self.vid_picker_changed)
        self.label_h5_picker.changed.connect(self.convert_h5_todict)
        label_txt_picker.changed.connect(self.convert_txt_todict)
        self.ind_spinbox.changed.connect(self.individual_changed)
        self.spinbox.changed.connect(self.behaviour_changed)
        push_button.changed.connect(self.save_classification_data)

        self.ind = 0
        self.dlc_data = None
        self.behaviour_no = 0
        self.clean = None  # old function may be useful in future
        self.im_subset = None
        # self.points = None
        self.labeled = False
        self.behaviours = []
        self.choices = []
        self.b_labels = None
        self.decoder_data_dir = None
        self.ground_truth_ethogram = None
        self.ethogram = None
        self.regions_layer = None
        self.points_layer = None
        self.track_layer = None
        self.detection_layer = None
        self.regions = []
        self.points = None
        self.tracks = None

        self.add_1d_widget()
        self.viewer.dims.events.current_step.connect(self.update_slider)

        ### infererence
        self.batch_size_spinbox = SpinBox(label="Batch Size", value=16, max = 1024)
        self.num_workers_spinbox = SpinBox(label="Num Workers", value=8)
        self.lr_spinbox = FloatSpinBox(
            label="Learning Rate", value=0.01, step=0.0000
        )
        self.dropout_spinbox = FloatSpinBox(label="Dropout", value=0)
        self.num_labels_spinbox = SpinBox(label="Num labels", value=0)
        self.num_channels_spinbox = SpinBox(label="Num channels", value=3)

        self.model_dropdown = ComboBox(
            label="Model type",
            choices=["Detection", "PoseEstimation", "BehaviourDecode"],
            tooltip="Select model for predicting behaviour",
            value="BehaviourDecode",
        )

        self.chkpt_dropdown = ComboBox(
            label="Checkpoint",
            choices=[],
            tooltip="Select chkpt for predicting behaviour",
        )
        self.model_dropdown.changed.connect(self.populate_chkpt_dropdown)

        self.live_checkbox = CheckBox(text="Live Decode")
        self.live_checkbox.clicked.connect(self.live_decode)

        self.analyse_button = PushButton(label="Analyse")
        self.analyse_button.clicked.connect(self.analyse)

        self.train_button = PushButton(label="Train")
        self.train_button.clicked.connect(self.train)

        self.finetune_button = PushButton(label="Finetune")
        self.finetune_button.clicked.connect(self.finetune)

        self.test_button = PushButton(label="Test")
        self.test_button.clicked.connect(self.test)

        self.clear_button = PushButton(label="Clear")
        self.clear_button.clicked.connect(self.full_reset)

        self.extend(
            [
                self.batch_size_spinbox,
                self.lr_spinbox,
                self.model_dropdown,
                self.chkpt_dropdown,
                self.train_button,
                self.live_checkbox,
                self.analyse_button,
                self.finetune_button,
                self.test_button,
                self.clear_button,
            ]
        )  # self.num_workers_spinbox,  self.dropout_spinbox,
        # self.num_labels_spinbox, self.num_channels_spinbox, self.model_dropdown,,
        # ])

    def full_reset(self):
        self.ind = 0
        self.ind_spinbox.value = 0

        self.behaviour_no = 0
        self.spinbox.value = 0
        self.spinbox.max = 0

        self.classification_data = {}
        self.point_subset = np.array([])

        self.coords_data = {}

        self.ind = 0
        self.ind_spinbox.max = 0
        self.dlc_data = None

        self.behaviour_no = 0
        self.clean = None  # old function may be useful in future
        self.im_subset = None
        self.im = None
        self.video_file = None
        self.vid_picker.value = ""
        self.labeled = False
        self.behaviours = []

        self.b_labels = None

        self.ground_truth_ethogram = None
        self.ethogram = None
        self.regions_layer = None
        self.points_layer = None
        self.track_layer = None
        self.detection_layer = None
        self.regions = []
        self.points = None
        self.tracks = None
        self.zebdata = None

        ## reset layers
        self.reset_layers()

        self.reset_viewer1d_layers()
        self.add_frame_line()
        self.label_menu.choices = self.choices
        try:
            self.populate_chkpt_dropdown()
        except:
            pass

    def decoder_dir_changed(self, value):
        # Look for and load yaml configuration file
        # load config
        self.full_reset()
        self.decoder_data_dir = value
        print(f"Decoder Data Folder is {self.decoder_data_dir}")
        self.config_file = os.path.join(
            self.decoder_data_dir, "decoder_config.yml"
        )

        try:
            with open(self.config_file) as file:
                self.config_data = yaml.safe_load(file)

            n_nodes = self.config_data["data_cfg"]["V"]
            center_node = self.config_data["data_cfg"]["center"]
            self.set_center_node(center_node)
            self.set_n_nodes(n_nodes)
            self.classification_dict = self.config_data["data_cfg"][
                "classification_dict"
            ]
            self.choices = [v for v in self.classification_dict.values()]
            self.label_menu.choices = self.choices

            try:
                self.dataset = self.config_data["data_cfg"]["dataset"]
            except:
                self.dataset = None

        except:
            print("No configuration yaml located in decoder data folder")

        # check lightning_logs folder exists - if not create it
        if (
            os.path.exists(
                os.path.join(self.decoder_data_dir, "lightning_logs")
            )
            == False
        ):
            # make this directory
            os.mkdir(os.path.join(self.decoder_data_dir, "lightning_logs"))

        self.populate_chkpt_dropdown()  # load ckpt files if any
        self.initialise_params()

        print(f"decoder config is {self.config_data}")
        self.view_data()

    def populate_chkpt_dropdown(self, event=None):
        # get all checkpoint files and allow user to select one

        if (event == None) | (event == "BehaviourDecode"):
            log_folder = os.path.join(self.decoder_data_dir, "lightning_logs")
            if os.path.exists(log_folder):
                version_folders = [
                    version_folder
                    for version_folder in os.listdir(log_folder)
                    if "version" in version_folder
                ]

                self.ckpt_files = []
                for version_folder in version_folders:
                    version_folder_files = os.listdir(
                        os.path.join(log_folder, version_folder)
                    )

                    for sub_file in version_folder_files:
                        if ".ckpt" in sub_file:
                            ckpt_file = os.path.join(version_folder, sub_file)
                            self.ckpt_files.append(ckpt_file)

            else:
                self.ckpt_files = []

        elif event == "Detection":
            self.ckpt_files = []

        elif event == "PoseEstimation":
            self.ckpt_files = []

        self.chkpt_dropdown.choices = self.ckpt_files

    def update_slider(self, event):
        print("updating slider")
        # if (event.axis == 0):
        print(event)
        self.frame = event.value[0]
        # try:

        if self.behaviour_no > 0:
            try:
                add_frame = self.behaviours[self.behaviour_no - 1][0]
            except:
                print("updating slider from classification data")
                add_frame = self.classification_data[self.ind][
                    self.behaviour_no
                ][
                    "start"
                ]  # self.behaviours[self.behaviour_no - 1][0]

        else:
            add_frame = 0
        self.frame_line.data = np.c_[
            [self.frame + add_frame, self.frame + add_frame],
            [0, self.frame_line.data[1, 0]],
        ]

        # except:
        #    print("Failed to update frame line")

        print(f"updating slider frame {self.frame}")

        if self.live_checkbox.value:
            # create behaviour from points
            # pass to mode
            # logits and add to a ethogram in viewer1d
            exists = False
            if self.model_dropdown.value == "Detection":
                if self.detection_layer == None:
                    labels = ["0"]
                    properties = {
                        "label": labels,
                    }

                    # text_params = {
                    #    "text": "label: {label}",
                    #    "size": 12,
                    #    "color": "green",
                    #    "anchor": "upper_left",
                    #    "translation": [-3, 0],
                    #    }

                    self.detection_layer = self.viewer.add_shapes(
                        np.zeros((1, 4, 3)),
                        shape_type="rectangle",
                        edge_width=self.im_subset.data.shape[2] / 200,
                        edge_color="#55ff00",
                        face_color="transparent",
                        visible=True,
                        properties=properties,
                        # text = text_params,
                    )

                # check frame data already exists

                for shape in self.detection_layer.data:  # its a list
                    if shape[0, 0] == self.frame:
                        print("bbox already exists")
                        exists = True
                        break

                if exists == False:
                    labels = self.detection_layer.properties["label"].tolist()
                    shape_data = self.detection_layer.data

                    print(f"shape data shape is {len(shape_data)}")
                    print(f"labels are {labels}")

                    if self.detection_backbone == "YOLOv8":
                        h = self.im_subset.data[self.frame].shape[0]
                        results = self.model(
                            self.im_subset.data[self.frame], imgsz=h - (h % 32)
                        )
                        for result in results:
                            names = result.names
                            boxes = result.boxes
                            for box in boxes:
                                if box.conf > 0.1:
                                    label = names[int(box.cls.cpu().numpy())]
                                    labels.append(label)
                                    (
                                        x_min,
                                        y_min,
                                        x_max,
                                        y_max,
                                    ) = box.xyxy.cpu().numpy()[0]
                                    new_shape = np.array(
                                        [
                                            [self.frame, y_min, x_min],
                                            [self.frame, y_min, x_max],
                                            [self.frame, y_max, x_max],
                                            [self.frame, y_max, x_min],
                                        ]
                                    )

                                    shape_data.append(new_shape)

                    elif self.detection_backbone == "YOLOv5":
                        results = self.model(self.im_subset.data[self.frame])
                        result_df = results.pandas().xyxy[0]
                        print(result_df)

                        result_df = self.remove_overlapping_bboxes(result_df)
                        print(result_df)

                        for row in result_df.index:
                            labels.append(result_df.loc[row, "name"])

                            x_min = result_df.loc[row, "xmin"]
                            x_max = result_df.loc[row, "xmax"]
                            y_min = result_df.loc[row, "ymin"]
                            y_max = result_df.loc[row, "ymax"]

                            new_shape = np.array(
                                [
                                    [self.frame, y_min, x_min],
                                    [self.frame, y_min, x_max],
                                    [self.frame, y_max, x_max],
                                    [self.frame, y_max, x_min],
                                ]
                            )

                            shape_data.append(new_shape)
                            # shape_data = np.array(shape_data)

                    print(labels)
                    self.detection_layer.data = shape_data
                    self.detection_layer.properties = {"label": labels}

            elif self.model_dropdown.value == "PoseEstimation":
                # check frame data already exists
                if self.points_layer is None:
                    point_properties = {"confidence": [0], "ind": [0]}
                    self.points_layer = self.viewer.add_points(
                        np.zeros((1, 3)),
                        properties=point_properties,
                        size=self.im_subset.data.shape[2] / 100,
                    )

                for point in self.points_layer.data.tolist():  # its a list
                    if point[0] == self.frame:
                        print("point exists")
                        exists = True
                        break

                if exists == False:
                    im = self.im_subset.data[self.frame]

                    person_results = []

                    if self.detection_layer is not None:
                        for shape in self.detection_layer.data:  # its a list
                            if shape[0, 0] == self.frame:
                                print(shape)
                                L = shape[0, 2]  # xmin
                                T = shape[0, 1]  # ymin
                                R = shape[2, 2]  # xmax
                                B = shape[2, 1]  # ymax

                                bbox_data = {
                                    "bbox": (L, T, R, B),
                                    "track_id": len(person_results) + 1,
                                }

                                person_results.append(bbox_data)

                    if len(person_results) > 0:
                        pose_results, _ = inference_top_down_pose_model(
                            self.model,
                            im,
                            person_results=person_results,
                            format="xyxy",
                        )
                        print(pose_results)

                        points = []

                        point_properties = self.points_layer.properties.copy()

                        for ind in range(len(pose_results)):
                            keypoints = pose_results[ind]["keypoints"]
                            for ncoord in range(keypoints.shape[0]):
                                x, y, ci = keypoints[ncoord]

                                points.append((self.frame, y, x))
                                point_properties["confidence"] = np.append(
                                    point_properties["confidence"], ci
                                )
                                point_properties["ind"] = np.append(
                                    point_properties["ind"], ind
                                )

                        print(point_properties)
                        print(points)
                        self.points_layer.data = np.concatenate(
                            (self.points_layer.data, np.array(points))
                        )

                        self.points_layer.properties[
                            "confidence"
                        ] = point_properties["confidence"]
                        self.points_layer.properties["ind"] = point_properties[
                            "ind"
                        ]
                        print(self.points_layer.properties)

                        # check for bounding boxes
                        # call inference_top_down_mode(self.model, im)

            elif self.model_dropdown.value == "BehaviourDecode":
                denominator = self.config_data["data_cfg"]["denominator"]
                T_method = self.config_data["data_cfg"]["T"]
                fps = self.config_data["data_cfg"]["fps"]

                if T_method == "window":
                    T = 2 * int(fps / denominator)

                elif type(T_method) == "int":
                    T = T_method  # these methods assume behaviours last the same amount of time -which is a big assumption

                elif T_method == "None":
                    T = 43
                self.behaviours = [
                    (self.frame + n, self.frame + n + T)
                    for n in range(self.batch_size)
                ]
                # check if frame already processed
                if self.ethogram.data[:, self.frame].sum() == 0:
                    self.preprocess_bouts()

                    model_input = self.zebdata[: self.batch_size][0].to(
                        self.device
                    )
                    with torch.no_grad():
                        probs = self.model(model_input).cpu().numpy()

                    self.ethogram.data[
                        :, self.frame : self.frame + self.batch_size
                    ] = probs.T
                    # have to switch the layer off and on for chnage to be seen
                    self.ethogram.visible = False
                    self.ethogram.visible = True
                    # self.viewer1d.reset_view()
                    print(f"Probs are {probs}")
                else:
                    print("Frame already processed")

    # def extract_behaviour_from_frame(self):
    #    T = 43 # assign this better in future
    #    54t

    def add_1d_widget(self):
        self.viewer1d = napari_plot.ViewerModel1D()  # ViewerModel1D()
        widget = QtViewer(self.viewer1d)

        self.viewer.window.add_dock_widget(
            widget, area="bottom", name="Movement"
        )
        self.viewer1d.axis.x_label = "Time"
        self.viewer1d.axis.y_label = "Movement"
        self.viewer1d.reset_view()
        self.frame = 0
        self.add_frame_line()

    def add_frame_line(self):
        self.frame_line = self.viewer1d.add_line(
            np.c_[[self.frame, self.frame], [0, 10]],
            color="gray",
            name="Frame",
        )

        # Moving frames? - redundant
        # Preprocess_txt_file? - maybe include later if a tx file is selected with a pop up?
        # Extend window - redundant

    def add_behaviour_from_selected_area(self, value):
        # get x range of viewer1d
        # subset data using those frame indices
        # append behaviour to self behaviours

        start, stop = self.viewer1d.camera.rect[:2]
        self.behaviours.append((int(start), int(stop)))
        print(self.behaviours)

        if self.ind not in list(self.classification_data.keys()):
            self.classification_data[self.ind] = {}

        self.spinbox.max = len(self.behaviours)
        self.spinbox.value = len(self.behaviours)
        # self.behaviour_changed(len(self.behaviours))

    def plot_movement_1d(self):
        # plot colors mapped to confidence interval - can't do this yet even for scatter
        # ci = self.ci.iloc[:].std(axis = 0).to_numpy()
        # norm = plt.Normalize()
        # colors = plt.cm.jet(norm(ci))
        choices = self.label_menu.choices
        t = np.arange(self.gauss_filtered.shape[0])

        self.viewer1d.add_line(
            np.c_[t, self.gauss_filtered],
            color="magenta",
            label="Movement",
            name="Movement",
        )
        thresh = np.median(self.gauss_filtered) + self.threshold
        self.viewer1d.add_line(
            np.c_[[0, self.gauss_filtered.shape[0]], [thresh, thresh]],
            color="cyan",
            label="Movement threshold",
            name="Threshold",
        )
        self.viewer1d.reset_view()
        self.label_menu.choices = choices
        self.frame_line.data = np.c_[
            [self.frame, self.frame], [0, thresh + (0.5 * thresh)]
        ]

    def plot_behaving_region(self):
        self.regions.append(([self.start, self.stop], "vertical"))
        print(self.regions)
        choices = self.label_menu.choices
        # regions = [
        #    ([self.start, self.stop], "vertical"),
        # ]
        if self.regions_layer is None:
            self.regions_layer = self.viewer1d.add_region(
                self.regions,
                color=["green"],
                opacity=0.4,
                name="Behaviour",
            )
        else:
            self.regions_layer.data = self.regions
        self.label_menu.choices = choices

    def reset_viewer1d_layers(self):
        try:
            # self.viewer1d.clear_canvas()
            for layer in self.viewer1d.layers:
                # print(layer)
                self.viewer1d.layers.remove(layer)
            self.viewer1d.clear_canvas()
            print(f"Layers remaining are {self.viewer1d.layers}")
        except:
            pass

    def reset_layers(self):
        """Resest all napari layers. Called three times to ensure layers removed."""
        for layer in reversed(self.viewer.layers):
            # print(layer)
            self.viewer.layers.remove(layer)
        time.sleep(1)
        print(f"Layers remaining are {self.viewer.layers}")
        try:
            for layer in self.viewer.layers:
                # print(layer)
                self.viewer.layers.remove(layer)
        except:
            pass

        try:
            for layer in self.viewer.layers:
                # print(layer)
                self.viewer.layers.remove(layer)
        except:
            pass

    def save_current_data(self):
        """Called when behaviour is changed."""
        self.classification_data[self.ind][self.last_behaviour] = {
            "classification": self.label_menu.current_choice,
            "coords": self.point_subset,
            "start": self.start,
            "stop": self.stop,
            "ci": self.ci_subset,
        }

        etho = self.classification_data_to_ethogram()
        self.populate_groundt_etho(etho)

    def update_classification(self):
        """Updates classification label in GUI"""
        print("updated")
        # if self.labeled:
        #    try:
        #        self.label_menu.choices = tuple(self.txt_behaviours)
        #    except:
        #        pass
        # print(self.label_menu.choices)
        # print(self.classification_data[self.ind][self.behaviour_no]["classification"])
        try:
            print(self.label_menu.choices)
            print(
                self.classification_data[self.ind][self.behaviour_no][
                    "classification"
                ]
                in self.label_menu.choices
            )
            print(
                self.classification_data[self.ind][self.behaviour_no][
                    "classification"
                ]
            )
            print(
                type(
                    self.classification_data[self.ind][self.behaviour_no][
                        "classification"
                    ]
                )
            )
            self.label_menu.value = self.classification_data[self.ind][
                self.behaviour_no
            ]["classification"]
        except:
            self.label_menu.value = str(
                self.classification_data[self.ind][self.behaviour_no][
                    "classification"
                ]
            )

    def update_extract_method(self, value):
        self.extract_method = value
        print(f"Extract method is {self.extract_method}")

    def get_points(self):
        """Converts coordinates into points format for napari points layer"""
        # print("Getting Individuals Points")
        x_flat = self.x.to_numpy().flatten()
        y_flat = self.y.to_numpy().flatten()

        try:
            z_flat = self.z.to_numpy().flatten()

        except:
            print("no z frame coord")
            z_flat = np.tile(self.x.columns, self.x.shape[0])

        zipped = zip(z_flat, y_flat, x_flat)
        points = [[z, y, x] for z, y, x in zipped]
        points = np.array(points)

        self.points = points

    def get_tracks(self):
        """Converts coordinates into tracks format for napari tracks layer"""
        # print("Getting Individuals Tracks")

        x_nose = self.x.to_numpy()[self.center_node]  # -1 # change this to tail node
        y_nose = self.y.to_numpy()[self.center_node]  # -1

        z_nose = np.arange(self.x.shape[1])
        nose_zipped = zip(z_nose, y_nose, x_nose)
        tracks = np.array([[0, z, y, x] for z, y, x in nose_zipped])

        self.tracks = tracks

    def egocentric_variance(self):
        """Estimates locomotion based on peaks of egocentric movement ."""
        reshap = self.points.reshape(self.n_nodes, -1, 3)
        center = reshap[self.center_node, :, 1:]  # selects x,y center nodes
        self.egocentric = reshap.copy()
        self.egocentric[:, :, 1:] = reshap[:, :, 1:] - center.reshape(
            (-1, *center.shape)
        )  # subtract center nodes
        absol_traj = (
            self.egocentric[:, 1:, 1:] - self.egocentric[:, :-1, 1:]
        )  # trajectory
        self.euclidean = np.sqrt(
            np.abs((absol_traj[:, :, 0] ** 2) + (absol_traj[:, :, 1] ** 2))
        )  # euclidean trajectory
        var = np.median(self.euclidean, axis=0)  # median movement
        self.gauss_filtered = gaussian_filter1d(
            var, int(self.fps / 10)
        )  # smoothed movement
        amd = np.median(self.gauss_filtered - self.gauss_filtered[0]) / 0.6745
        peaks = find_peaks(
            self.gauss_filtered,
            prominence=amd * 7,
            distance=int(self.fps / 2),
            width=5,
            rel_height=0.6,
        )  # zeb

        # check stop does not come before start
        self.behaviours = [
            (int(start) - 20, int(end) + 20)
            for start, end in zip(peaks[1]["left_ips"], peaks[1]["right_ips"])
            if end > start
        ]  # zeb

        # check behaviour has high confidence score
        self.behaviours = [
            (start, end)
            for start, end in self.behaviours
            if self.check_behaviour_confidence(start, end)
        ]

        # check no overlap
        b_arr = np.array(self.behaviours)
        # b_arr = (b_arr/30) #convert to seconds
        overlap = b_arr[1:, 0] - b_arr[:-1, 1]
        overlap = np.where(overlap <= 0)[0] + 1

        b_arr[overlap, 0] = b_arr[overlap, 0] + 10
        b_arr[overlap - 1, 1] = b_arr[overlap, 0] - 10
        self.behaviours = b_arr.tolist()

        # self.moving_fig, self.moving_ax = plt.subplots()
        # self.moving_ax.plot(self.gauss_filtered)
        # self.moving_ax.scatter(peaks[0], self.gauss_filtered[peaks[0]])
        # [self.moving_ax.axvspan(int(start), int(end), color=(0, 1, 0, 0.5)) for start, end in self.behaviours]

    def calulate_orthogonal_variance(
        self, amd_threshold=2, confidence_threshold=0.8
    ):
        """Estimates locomotion based on orthogonal movement. Good for zebrafish."""
        # print("Calculating Orthogonal Variance")

        # Get euclidean trajectory - not necessary for orthogonal algorithm but batch requires it
        reshap = self.points.reshape(self.n_nodes, -1, 3)
        center = reshap[self.center_node, :, 1:]  # selects x,y center nodes
        self.egocentric = reshap.copy()
        self.egocentric[:, :, 1:] = reshap[:, :, 1:] - center.reshape(
            (-1, *center.shape)
        )  # subtract center nodes
        absol_traj = (
            self.egocentric[:, 1:, 1:] - self.egocentric[:, :-1, 1:]
        )  # trajectory
        self.euclidean = np.sqrt(
            np.abs((absol_traj[:, :, 0] ** 2) + (absol_traj[:, :, 1] ** 2))
        )  # euclidean trajectory

        # use egocentric instead to eliminate crop jitter
        # subsize = int(self.points.shape[0]/self.n_nodes)
        projections = []
        # maybe check % is 0
        for n in range(self.n_nodes):
            # subset = self.points[n*subsize: (n+1)*subsize]
            # trajectory_matrix = subset[1:, 1:] - subset[:-1, 1:]
            trajectory_matrix = absol_traj[n]
            orth_matrix = np.flip(trajectory_matrix, axis=1)
            orth_matrix[:, 0] = -orth_matrix[
                :, 0
            ]  # flip elements in trajectory matrix so x is y and y is x and reverse sign of first element. Only works for 2D vectors
            future_trajectory = trajectory_matrix[
                1:,
            ]  # shift trajectory by looking forward
            present_orth = orth_matrix[:-1,]  # subset all orth but last one
            projection = np.abs(
                (np.sum(future_trajectory * present_orth, axis=1))
                / np.linalg.norm(present_orth, axis=1)
            )  # project the dot product of each trajectory vector onto its orth vector
            projection[np.isnan(projection)] = 0
            projections.append(projection)

        proj = np.array(projections)
        var = np.median(proj, axis=0)
        self.gauss_filtered = gaussian_filter1d(
            var, int(self.fps / 10)
        )  # smoothed movement
        amd = st.median_abs_deviation(
            self.gauss_filtered
        )  #    np.median(self.gauss_filtered)/0.6745
        median = np.median(self.gauss_filtered)
        self.threshold = amd * amd_threshold
        peaks = find_peaks(
            self.gauss_filtered,
            prominence=self.threshold,
            distance=int(self.fps / 2),
            width=5,
            rel_height=0.6,
        )  # zeb

        # check stop does not come before start
        self.behaviours = [
            (int(start) - 20, int(end) + 20)
            for start, end in zip(peaks[1]["left_ips"], peaks[1]["right_ips"])
            if end > start
        ]  # zeb

        # check behaviour has high confidence score
        self.behaviours = [
            (start, end)
            for start, end in self.behaviours
            if self.check_behaviour_confidence(
                start, end, confidence_threshold
            )
        ]
        self.bad_behaviours = [
            (start, end)
            for start, end in self.behaviours
            if not self.check_behaviour_confidence(
                start, end, confidence_threshold
            )
        ]
        # check no overlap
        b_arr = np.array(self.behaviours)

        if b_arr.ndim == 2:
            # b_arr = (b_arr/30) #convert to seconds
            overlap = b_arr[1:, 0] - b_arr[:-1, 1]
            overlap = np.where(overlap <= 0)[0] + 1

            b_arr[overlap, 0] = b_arr[overlap, 0] + 10
            b_arr[overlap - 1, 1] = b_arr[overlap, 0] - 10
        self.behaviours = b_arr.tolist()

        # self.moving_fig, self.moving_ax = plt.subplots()
        # self.moving_ax.plot(self.gauss_filtered)
        # self.moving_ax.scatter(peaks[0], self.gauss_filtered[peaks[0]])
        # [self.moving_ax.axvspan(int(start), int(end), color=(0, 1, 0, 0.5)) for start, end in self.behaviours] # maybe utilise napari plot here?

    def check_behaviour_confidence(
        self, start, stop, confidence_threshold=0.8
    ):
        # subset confidence interval data for behaviour
        subset = self.ci.iloc[:, start:stop]

        # count number of values below threshold
        low_ci_counts = subset[(subset < confidence_threshold)].count()

        # average counts
        mean_low_ci_count = low_ci_counts.mean()

        # return boolean, True if ci counts are low (< 1) or high if ci_counts >1
        return mean_low_ci_count <= 1

    def plot_movement(self):
        """Plot movement as track in shape I, Z, Y, X.
        X is range 0 - 1000
        y is range 1250 - 1200

        """
        z_no = len(self.gauss_filtered)
        x = np.arange(z_no)
        ratio = 1000 / z_no
        x = (x * ratio).astype("int64")
        y = self.gauss_filtered  # scale y to within 50
        y_ratio = y.max() / 400
        y = -(y / y_ratio) + 200

        z = np.arange(0, z_no)
        i = np.zeros(z_no)
        self.movement = np.stack([i, z, y, x]).transpose()

        self.movement_layer = self.viewer.add_tracks(
            self.movement,
            tail_length=1000,
            tail_width=3,
            opacity=1,
            colormap="twilight",
        )
        self.label_menu.choices = self.choices

    def movement_labels(self):
        # get all moving frames
        moving_frames_idx = np.array([], dtype="int64")

        for start, stop in np.array(
            self.behaviours
        ).tolist():  # [random_integers].tolist():
            arr = np.arange(start, stop, dtype="int64")
            moving_frames_idx = np.append(moving_frames_idx, arr)

        # get centre node
        centre = self.points.reshape(self.n_nodes, -1, 3)[self.center_node]

        # tile and reshape centre location
        centre_rs = np.tile(centre[moving_frames_idx], 4).reshape(-1, 4, 3)

        # create array to add to centre node to create bounding box
        add_array = np.array(
            [[0, -100, -100], [0, -100, 100], [0, 100, 100], [0, 100, -100]]
        )

        # define boxes by adding to centre_rs
        boxes = centre_rs + add_array.reshape(-1, *add_array.shape)

        # specify label params
        nframes = 300  # moving_frames_idx.shape[
        # 0
        # ]   at the moment more than 300 is really slow
        labels = ["movement"] * nframes
        properties = {
            "label": labels,
        }

        # specify the display parameters for the text
        text_params = {
            "text": "label: {label}",
            "size": 12,
            "color": "green",
            "anchor": "upper_left",
            "translation": [-3, 0],
        }

        # add shapes layer
        self.shapes_layer = self.viewer.add_shapes(
            boxes[:nframes],
            shape_type="rectangle",
            edge_width=self.im_subset.data.shape[2] / 200,
            edge_color="#55ff00",
            face_color="transparent",
            visible=True,
            properties=properties,
            text=text_params,
            name="Movement",
        )
        self.label_menu.choices = self.choices

    def h5_picker_changed(self, event):
        """This function is called when a new h5/csv from DLC is selected.

        Parameters:

        event: widget event"""
        print(f"DLC File Changed to {event}")
        try:
            self.h5_file = event.value.value
        except:
            try:
                self.h5_file = event.value
            except:
                self.h5_file = str(event)
        self.full_reset()

        self.read_coords(self.h5_file)

        self.populate_chkpt_dropdown()  # because it keeps erasing it

    def vid_picker_changed(self, event):
        """This function is called when a new video is selected.

        Parameters:

        event: widget event"""
        print(f"Video File Changed to {event}")

        try:
            self.video_file = event.value.value
        except:
            try:
                self.video_file = event.value
            except:
                self.video_file = str(event)

        # check avi, mp4
        if (".avi" in self.video_file.lower()) | (
            ".mp4" in self.video_file.lower()
        ):
            # vid = pims.open(str(self.video_file))
            self.fps = self.config_data["data_cfg"]["fps"]
            try:
                self.im = VideoReaderNP(str(self.video_file))
            except:
                print("Couldn't read video file")
                self.im = None

            if self.im is not None:
                # add a video layer if none
                if self.im_subset is None:
                    self.im_subset = self.viewer.add_image(
                        self.im, name="Video Recording"
                    )
                    self.label_menu.choices = self.choices
                else:
                    self.im_subset.data = self.im

                self.populate_chkpt_dropdown()  # because adding layers keeps erasing it

    def convert_h5_todict(self, event):
        """reads pytables and converts to dict. If new dict saved overwrites existing pytables"""
        try:
            self.labeled_h5 = event.value.value
        except:
            try:
                self.labeled_h5 = event.value
            except:
                self.labeled_h5 = str(event)

        self.labeled_h5_file = tb.open_file(self.labeled_h5, mode="a")
        self.classification_data = {}

        for group in self.labeled_h5_file.root.__getattr__("_v_groups"):
            ind = self.labeled_h5_file.root[group]
            behaviour_dict = {}
            arrays = {}

            for array in self.labeled_h5_file.list_nodes(
                ind, classname="Array"
            ):
                arrays[int(array.name)] = array
            tables = []

            for table in self.labeled_h5_file.list_nodes(
                ind, classname="Table"
            ):
                tables.append(table)

            behaviours = []
            classifications = []
            starts = []
            stops = []
            cis = []
            for row in tables[0].iterrows():
                behaviours.append(row["number"])
                classifications.append(row["classification"])
                starts.append(row["start"])
                stops.append(row["stop"])

            for behaviour, (classification, start, stop) in enumerate(
                zip(classifications, starts, stops)
            ):
                class_dict = {
                    "classification": classification.decode("utf-8"),
                    "coords": arrays[behaviour + 1][:, :3],
                    "start": start,
                    "stop": stop,
                    "ci": arrays[behaviour + 1][:, 3],
                }
                behaviour_dict[behaviour + 1] = class_dict
            self.classification_data[int(group)] = behaviour_dict

        self.labeled = True
        self.labeled_h5_file.close()
        self.individual_changed(self.ind)  # reload ind data
        self.ind_spinbox.max = max(self.classification_data.keys())
        # self.ind_spinbox.value = 0
        self.spinbox.value = 0

        self.tracks = None  # set this to none as it's not saved
        # add self behaviours

        # self.ind = 0

        # self.choices = pd.Series([label["classification"] for k,label in self.classification_data[1].items()]).unique().tolist()
        # print(self.choices)
        # self.label_menu.choices = tuple(self.choices)

    def convert_oft_todict(self, event):
        try:
            self.labeled_txt = event.value.value
        except:
            try:
                self.labeled_txt = event.value
            except:
                self.labeled_txt = str(event)

        event_df = pd.read_csv(self.labeled_txt)  ## no header

        self.labeled = True

        key = list(self.coords_data.keys())[self.ind - 1]
        self.x = self.coords_data[key]["x"]
        self.y = self.coords_data[key]["y"]
        self.ci = self.coords_data[key]["ci"]
        self.get_points()

        ind_dict = {}
        for n in range(event_df.shape[0]):
            row = event_df.iloc[n]
            self.start = int(row.start * self.fps)  # Time
            self.stop = int(row.end * self.fps)  # Duration
            classification = row.label  # TrackName

            self.point_subset = self.points.reshape((self.n_nodes, -1, 3))[
                :, int(self.start) : int(self.stop)
            ].reshape(-1, 3)
            self.point_subset = self.point_subset - np.array(
                [self.start, 0, 0]
            )
            self.ci_subset = (
                self.ci.iloc[:, int(self.start) : int(self.stop)]
                .to_numpy()
                .flatten()
            )
            behav_dic = {
                "classification": classification,
                "coords": self.point_subset,
                "start": self.start,
                "stop": self.stop,
                "ci": self.ci_subset,
            }

            ind_dict[n + 1] = behav_dic

        self.classification_data = {}
        self.classification_data[1] = ind_dict  # assuming 1 individual
        print(self.classification_data.keys())

        self.ind_spinbox.max = max(self.classification_data.keys())
        self.ind_spinbox.value = 1
        self.spinbox.value = 0
        self.behaviour_no = 0

        self.populate_chkpt_dropdown()
        self.label_menu.choices = self.choices

        etho = self.classification_data_to_ethogram()
        self.populate_groundt_etho(etho)

        print(f"Loaded OFT txt file is {event_df}")
        # self.label_menu.reset_choices() # this should be set by the config
        # self.txt_behaviours = event_df.iloc[:, 2].unique().astype("str").tolist()
        # self.label_menu.reset_choices()
        # self.label_menu.choices = tuple(self.txt_behaviours)
        # print(self.label_menu.choices)

    def classification_data_to_ethogram(self):
        N = self.im.shape[0]  # self.dlc_data.shape[0]
        etho = np.zeros((len(self.label_dict), N))

        if len(self.classification_data.keys()) > 0:
            for bout, data in self.classification_data[self.ind].items():
                idx = np.arange(data["start"], data["stop"])
                try:
                    label = self.label_dict[data["classification"]]
                except:
                    # if label dict stores ints not strings
                    label = self.label_dict[int(data["classification"])]
                etho[label, idx] = 1

        return etho

    def populate_groundt_etho(self, etho):
        if self.ground_truth_ethogram is not None:
            self.ground_truth_ethogram.data = etho
            self.ground_truth_ethogram.visible = False
            self.ground_truth_ethogram.visible = True
        else:
            self.ground_truth_ethogram = self.viewer1d.add_image(
                etho, name="Ground truth", opacity=0.5
            )

    def populate_predicted_etho(self, etho):
        if self.ethogram is not None:
            self.ethogram.data = etho
            self.ethogram.visible = False
            self.ethogram.visible = True
        else:
            self.ethogram = self.viewer1d.add_image(
                etho, name="Predicted", opacity=0.5
            )

    def convert_txt_todict(self, event):
        """Reads event text file and converts it to usable format to display behaviours in GUI."""
        # self.full_reset()
        try:
            self.labeled_txt = event.value.value
        except:
            try:
                self.labeled_txt = event.value
            except:
                self.labeled_txt = str(event)

        # read txt file

        if "OFT" in self.dataset:
            self.convert_oft_todict(event)

        else:  # self.dataset == "Drosophila":
            event_df = pd.read_csv(self.labeled_txt, ",", header=2)

            if self.preprocess_txt_file:
                event_df = self.preprocess_txt(event_df)

            if self.extend_window:
                event_df.iloc[:, 1] = (
                    event_df.iloc[:, 1] + 500
                )  # added because maggot behaviour durations cut behaviour short
                event_df.iloc[:, :2] = (
                    (event_df.iloc[:, :2] / 1e3) * self.fps
                ).astype("int64")
            self.txt_behaviours = (
                event_df.iloc[:, 2].unique().astype("str").tolist()
            )
            # self.label_menu.choices = tuple(self.txt_behaviours)
            # fps = self.fps

            self.labeled = True
            self.ind_spinbox.value = 1

            key = list(self.coords_data.keys())[self.ind - 1]
            self.x = self.coords_data[key]["x"]
            self.y = self.coords_data[key]["y"]
            self.ci = self.coords_data[key]["ci"]
            self.get_points()

            ind_dict = {}
            for n, row in enumerate(event_df.itertuples()):
                self.start = int(row[1])  # Time
                self.stop = int(self.start + np.ceil(row[2]))  # Duration
                classification = row[3]  # TrackName

                self.point_subset = self.points.reshape((self.n_nodes, -1, 3))[
                    :, int(self.start) : int(self.stop)
                ].reshape(-1, 3)
                self.point_subset = self.point_subset - np.array(
                    [self.start, 0, 0]
                )
                self.ci_subset = (
                    self.ci.iloc[:, int(self.start) : int(self.stop)]
                    .to_numpy()
                    .flatten()
                )
                behav_dic = {
                    "classification": classification,
                    "coords": self.point_subset,
                    "start": self.start,
                    "stop": self.stop,
                    "ci": self.ci_subset,
                }

                ind_dict[n + 1] = behav_dic

            self.classification_data = {}
            self.classification_data[1] = ind_dict

            self.ind_spinbox.max = max(self.classification_data.keys())

            self.spinbox.value = 0
            self.behaviour_no = 0
            self.label_menu.reset_choices()
            self.txt_behaviours = (
                event_df.iloc[:, 2].unique().astype("str").tolist()
            )
            self.label_menu.reset_choices()
            self.label_menu.choices = tuple(self.txt_behaviours)
            print(self.label_menu.choices)

    def individual_changed(self, event):
        """Called when individual spin box is changed. Gets coordinates for new individual, adds a points and tracks layer
        to the GUI and also estimates periods of locomotion."""
        last_ind = self.ind
        self.ind = event
        print(f"New individual is individual {self.ind}")
        if self.ind > 0:
            # check ind in data
            if self.labeled == True:
                print(type(self.ind))
                # self.im_subset.data = self.im
                self.spinbox.max = len(
                    self.classification_data[self.ind].keys()
                )
                print(f"number of labelled behaviours is {self.spinbox.max}")
                self.label_menu.choices = self.choices
                self.populate_chkpt_dropdown()  # because keeps erasing dropdown choices

                # if len(self.behaviours) == 0: this would cover loading the class h5 file directly without extracting bouts
                # loop through class data and append start stops to self behaviours

            else:
                pass

            exists = len(
                [
                    n
                    for n, v in enumerate(self.coords_data)
                    if self.ind - 1 == n
                ]
            )
            if exists > 0:
                key = list(self.coords_data.keys())[self.ind - 1]

                self.x = self.coords_data[key]["x"]
                self.y = self.coords_data[key]["y"]
                self.ci = self.coords_data[key]["ci"]

                try:
                    self.z = self.coords_data[key]["z"]
                except:
                    print("no frame info")

                self.get_points()

                try:
                    self.get_tracks()
                except:
                    print("no tracks")
                # self.reset_layers()

                # self.viewer.add_image(self.im)
                self.im_subset.data = self.im

                # create points layer
                if self.points_layer is None:
                    # make point size a set ratio of window size
                    point_size = self.im_subset.data.shape[2] / 100

                    self.points_layer = self.viewer.add_points(
                        self.points, size=point_size, visible=True
                    )
                else:
                    self.points_layer.data = self.points

                # self.track_layer = self.viewer.add_tracks(self.tracks, tail_length = 100, tail_width = 3)
                self.label_menu.choices = self.choices
                self.populate_chkpt_dropdown()  # because keeps erasing dropdown choices

                # get egocentric
                reshap = self.points.reshape(self.n_nodes, -1, 3)
                center = reshap[
                    self.center_node, :, 1:
                ]  # selects x,y center nodes
                self.egocentric = reshap.copy()
                self.egocentric[:, :, 1:] = reshap[:, :, 1:] - center.reshape(
                    (-1, *center.shape)
                )  # subtract center nodes

            etho = self.classification_data_to_ethogram()  # assumes dlc data
            self.populate_groundt_etho(etho)

    def extract_behaviours(self, value=None):
        print(f"Extracting behaviours using {self.extract_method} method")
        # reset classification data
        # reset viewer1d

        self.reset_viewer1d_layers()

        if self.extract_method == "orth":
            # if (self.points.shape[0] > 1e6) & (cp.cuda.runtime.getDeviceCount() >0):
            #    print("Large video - sing GPU accelerated movement extraction")
            # self.calculate_orthogonal_variance_cupy()
            # else:
            self.amd_threshold = self.amd_threshold_spinbox.value
            self.confidence_threshold = self.confidence_threshold_spinbox.value
            self.calulate_orthogonal_variance(
                self.amd_threshold, self.confidence_threshold
            )
            self.movement_labels()
            # self.plot_movement()

        elif self.extract_method == "egocentric":
            self.egocentric_variance()
            self.movement_labels()
            # self.plot_movement()
        else:
            pass

        # check self behaviours doesnt have any where start is greater than end
        # if so remove them
        self.behaviours = [b for b in self.behaviours if b[0] < b[1]]

        # check if ind exists in classification data

        # exists = len([k for k in self.classification_data.keys() if k == self.ind])
        if self.ind in self.classification_data:  # exists > 0:
            # reset -covers cases where classification data needs to be overwritten
            self.classification_data[self.ind] = {}
        else:
            self.classification_data[self.ind] = {}

        # else:
        #     self.ind_spinbox.value = last_ind
        self.spinbox.value = 0
        self.spinbox.max = len(self.behaviours)
        self.plot_movement_1d()

        self.populate_chkpt_dropdown()

    def behaviours_to_classification_data(self):
        """Converts extracted behaviours to classification data format."""
        for n, (start, stop) in enumerate(self.behaviours):
            point_subset = self.points.reshape(
                    (self.n_nodes, -1, 3)
                )[:, start:stop].reshape(-1, 3)
            point_subset = point_subset - np.array(
                        [start, 0, 0]
                    )  # zero z because add_image has zeroed
            self.classification_data[self.ind][n + 1] = {
                "classification": "unclassified",
                "coords": point_subset,
                "start": start,
                "stop": stop,
                "ci": self.ci.iloc[:, start:stop].to_numpy().flatten(),
            }

    def behaviour_changed(self, event):
        """Called when behaviour number is changed."""
        self.last_behaviour = self.behaviour_no

        try:
            # choices = self.label_menu.choices
            self.viewer.layers.remove(self.shapes_layer)
            del self.shapes_layer

            # reset_choices as they seem to be forgotten when layers added or deleted
            self.label_menu.choices = self.choices

        except:
            print("no shape layer")

        try:
            self.detection_layer.visible = False
        except:
            print("no detection layer")

        try:
            self.behaviour_no = event.value

        except:
            self.behaviour_no = event

        print(f"New behaviour is {self.behaviour_no}")

        # if (self.labeled != True):

        #    self.spinbox.max = len(self.behaviours)

        if self.behaviour_no > 0:
            if self.ind in self.classification_data.keys():
                if (self.last_behaviour != 0) & (
                    self.behaviour_no != 0
                ):  # event.value > 1:
                    self.save_current_data()

                # exists = len([k for k in self.classification_data[self.ind].keys() if k == self.behaviour_no])
                if (
                    self.behaviour_no in self.classification_data[self.ind]
                ):  # exists > 0:
                    print("exists")
                    # use self.classification_data

                    # self.reset_layers()

                    # get points from here, too complicated to create tracks here i think
                    # print(self.label_menu.choices)
                    self.point_subset = self.classification_data[self.ind][
                        self.behaviour_no
                    ]["coords"]
                    self.start = self.classification_data[self.ind][
                        self.behaviour_no
                    ]["start"]
                    self.stop = self.classification_data[self.ind][
                        self.behaviour_no
                    ]["stop"]
                    self.ci_subset = self.classification_data[self.ind][
                        self.behaviour_no
                    ]["ci"]
                    # self.im_subset = self.viewer.add_image(self.im[self.start:self.stop])
                    # self.points_layer = self.viewer.add_points(self.point_subset, size=5)

                    self.im_subset.data = self.im[self.start : self.stop]

                    # self.im_subset = self.viewer.layers[0]
                    try:
                        self.points_layer.data = self.point_subset
                    except:
                        self.points_layer = self.viewer.add_points(
                            self.point_subset,
                            size=self.im_subset.data.shape[2] / 100,
                        )
                        self.label_menu.choices = self.choices
                    try:
                        if self.tracks is not None:
                            self.track_subset = self.tracks[
                                self.start : self.stop
                            ]
                            self.track_subset = self.track_subset - np.array(
                                [0, self.start, 0, 0]
                            )  # zero z because add_image has zeroed

                            try:
                                self.track_layer.data = self.track_subset
                            except:
                                self.track_layer = self.viewer.add_tracks(
                                    self.track_subset,
                                    tail_length=500,
                                    tail_width=3,
                                )
                                self.label_menu.choices = self.choices
                        # self.points_layer.data = self.point_subset
                    except:
                        print("No tracks")
                        self.populate_chkpt_dropdown()
                        if len(self.label_menu.choices) == 0:
                            self.label_menu.choices = self.choices
                    print(f"label menu choices are {self.label_menu.choices}")
                    print(type(self.label_menu.choices))
                    print(len(self.label_menu.choices))
                    print(self.choices)
                    if self.label_menu.choices == ():
                        try:
                            self.label_menu.choices = self.choices
                        except:
                            pass
                    self.update_classification()
                    # print(self.label_menu.choices)

                elif (
                    self.behaviour_no not in self.classification_data[self.ind]
                ) & (len(self.behaviours) > 0):
                    print("extracting behaviour")
                    self.start, self.stop = self.behaviours[
                        self.behaviour_no - 1
                    ]  # -1 because behaviours is array indexed
                    # self.reset_layers()

                    # self.im_subset = self.viewer.add_image(self.im[self.start:self.stop])
                    self.im_subset.data = self.im[self.start : self.stop]

                    dur = self.stop - self.start
                    self.point_subset = self.points.reshape(
                        (self.n_nodes, -1, 3)
                    )[:, self.start : self.stop].reshape(
                        (int(self.n_nodes * dur), 3)
                    )
                    # start_filter = self.points[:, 0] >= self.start
                    # end_filter = self.points[:, 0] < self.start
                    # self.point_subset = self.points[start_filter & end_filter]

                    self.point_subset = self.point_subset - np.array(
                        [self.start, 0, 0]
                    )  # zero z because add_image has zeroed
                    if self.tracks is not None:
                        self.track_subset = self.tracks[self.start : self.stop]
                        self.track_subset = self.track_subset - np.array(
                            [0, self.start, 0, 0]
                        )  # zero z because add_image has zeroed
                        try:
                            self.track_layer.data = self.track_subset
                        except:
                            self.track_layer = self.viewer.add_tracks(
                                self.track_subset,
                                tail_length=500,
                                tail_width=3,
                            )
                            self.label_menu.choices = self.choices

                    try:
                        self.ci_subset = (
                            self.ci.iloc[:, self.start : self.stop]
                            .to_numpy()
                            .flatten()
                        )
                    except:
                        print("ci is 1d")
                        self.ci_subset = self.ci.loc[
                            self.start : self.stop
                        ].to_numpy()
                    # self.im_subset = self.viewer.layers[0]
                    # self.im_subset.data = self.im[self.start:self.stop]

                    # self.im_subset = self.viewer.layers[0]
                    try:
                        self.points_layer.data = self.point_subset
                    except:
                        self.points_layer = self.viewer.add_points(
                            self.point_subset,
                            size=self.im_subset.data.shape[2] / 100,
                        )
                        self.label_menu.choices = self.choices

                    # self.points_layer.data = self.point_subset
                    # self.points_layer = self.viewer.add_points(self.point_subset, size=5)

                    if self.label_menu.choices == ():
                        print(self.label_menu.choices)
                        try:
                            self.label_menu.choices = self.txt_behaviours
                        except:
                            pass
                        print(self.label_menu.choices)

                    if self.b_labels is not None:
                        self.label_menu.value = self.b_labels[
                            self.behaviour_no - 1
                        ]
                        print(
                            f"Label score is {self.predictions.numpy()[self.behaviour_no-1]}"
                        )

                    self.plot_behaving_region()

                    # self.update_classification()
            elif len(self.behaviours) == 0:
                self.show_data(self.behaviour_no - 1)

        elif self.behaviour_no == 0:
            # restore full length
            print("restore full data")
            self.points_layer.data = self.points
            try:
                self.track_layer.data = self.tracks
            except:
                print("no tracks")
            self.im_subset.data = self.im

    def save_classification_data(self, event):
        try:
            self.save_to_h5(event)
        except:
            print("failed to save to h5")

        try:
            self.save_ethogram()
        except:
            print("failed to save to csv")

    def save_to_h5(self, event):
        """converts classification data to pytables format for efficient storage.
        Creates PyTables file, and groups for each individual. Classification is stored
        in a table and coordinates are stored in arrays"""
        filename = str(self.video_file) + "_classification.h5"
        if os.path.exists(filename):
            print(f"{filename} exists")
            classification_file = tb.open_file(
                filename, mode="a", title="classfication"
            )
        else:
            classification_file = tb.open_file(
                filename, mode="w", title="classfication"
            )

        # loop through ind, then loop through behaviours
        for ind in self.classification_data.keys():
            # if os.path.exists(filename): #add this to be able to append to h5 file
            #    print("classification file already exists")
            # ind_group =
            # ind_table =
            # else:
            ind_group = classification_file.create_group(
                "/", str(ind), "Individual" + str(ind)
            )
            ind_table = classification_file.create_table(
                ind_group,
                "labels",
                Behaviour,
                f"Individual {str(ind)} Behaviours",
            )

            ind_subset = self.classification_data[ind]

            for behaviour in ind_subset:
                try:
                    arr = ind_subset[behaviour]["coords"]
                    ci = ind_subset[behaviour]["ci"]
                    array = np.concatenate(
                        (arr, ci.reshape(-1, ci.shape[0]).T), axis=1
                    )
                    classification_file.create_array(
                        ind_group,
                        str(behaviour),
                        array,
                        "Behaviour" + str(behaviour),
                    )

                    ind_table.row["number"] = behaviour
                    ind_table.row["classification"] = ind_subset[behaviour][
                        "classification"
                    ]
                    ind_table.row["n_nodes"] = self.n_nodes
                    ind_table.row["start"] = ind_subset[behaviour]["start"]
                    ind_table.row["stop"] = ind_subset[behaviour]["stop"]
                    ind_table.row.append()
                    ind_table.flush()
                except:
                    print("no pose data")

        classification_file.close()

    def read_coords(self, h5_file):
        """Reads coordinates from DLC files (h5 and csv). Optional data cleaning."""

        if ".h5" in str(h5_file):
            self.dlc_data = pd.read_hdf(h5_file)
            data_t = self.dlc_data.transpose()

            try:
                data_t["individuals"]
                data_t = data_t.reset_index()
            except:
                data_t["individuals"] = ["individual1"] * data_t.shape[0]
                data_t = (
                    data_t.reset_index()
                    .set_index(
                        ["scorer", "individuals", "bodyparts", "coords"]
                    )
                    .reset_index()
                )

        if ".csv" in str(h5_file):
            self.dlc_data = pd.read_csv(h5_file, header=[0, 1, 2], index_col=0)
            data_t = self.dlc_data.transpose()
            data_t["individuals"] = ["individual1"] * data_t.shape[0]
            data_t = (
                data_t.reset_index()
                .set_index(["scorer", "individuals", "bodyparts", "coords"])
                .reset_index()
            )

        for individual in data_t.individuals.unique():
            if self.dataset == "OFT":
                bodypoints = [
                    "nose",
                    "headcentre",
                    "neck",
                    "earl",
                    "earr",
                    "bodycentre",
                    "bcl",
                    "bcr",
                    "hipl",
                    "hipr",
                    "tailbase",
                    "tailcentre",
                    "tailtip",
                ]
                print(f"Selecting bodypoints {bodypoints}")

                indv1 = data_t[
                    (data_t.individuals == individual)
                    & (data_t.bodyparts.isin(bodypoints))
                ]

            else:
                indv1 = data_t[data_t.individuals == individual].copy()

            # calculate interframe variability
            if self.clean:
                indv1.loc[:, 0:] = indv1.loc[:, 0:].interpolate(
                    axis=1
                )  # fillsna
            x = indv1.loc[indv1.coords == "x", 0:].reset_index(drop=True)
            y = indv1.loc[indv1.coords == "y", 0:].reset_index(drop=True)
            ci = indv1.loc[indv1.coords == "likelihood", 0:].reset_index(
                drop=True
            )

            # cleaning
            if self.clean:
                x[ci < 0.8] = np.nan
                y[ci < 0.8] = np.nan

                x = x.interpolate(axis=1)
                y = y.interpolate(axis=1)

            self.coords_data[individual] = {
                "x": x,
                "y": y,
                "ci": ci,
            }  # think i need ci for the model too
        self.ind_spinbox.max = int(data_t.individuals.unique().shape[0])

    def add_behaviour(self, value):
        behaviour_label = self.add_behaviour_text.value

        # assert value contains a word in string
        assert len(behaviour_label) > 0

        assert type(behaviour_label) == str
        choices = list(self.label_menu.choices)
        choices.append(behaviour_label)
        self.choices = choices
        self.label_menu.choices = tuple(choices)
        self.add_behaviour_text.value = ""

    def set_n_nodes(self, value):
        self.n_nodes = value
        print(f"Number of nodes is {self.n_nodes}")

    def set_center_node(self, value):
        self.center_node = value
        print(f"Center node is {self.center_node}")

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")

    def analyse(self, value):
        if self.model_dropdown.value == "BehaviourDecode":
            
            self.preprocess_bouts()  ## assumes behaviours extracted
            # create a classification data that has the same number as the number of behaviours
            self.behaviours_to_classification_data()
            self.predict_behaviours()
            self.update_classification_data_with_predictions()
            etho = self.classification_data_to_ethogram()
            self.populate_predicted_etho(etho)
            self.populate_chkpt_dropdown()
            self.save_classification_data(None)

        elif self.model_dropdown.value == "Detection":
            self.predict_object_detection()

        elif self.model_dropdown.value == "PoseEstimation":
            self.predict_poses()

    def predict_poses(self):
        try:
            from mmpose.apis import (
                init_pose_model,
                inference_top_down_pose_model,
            )
        except:
            print("mmpose not installed")
        self.initialise_params()

        if self.pose_config is not None:
            self.model = init_pose_model(self.pose_config, self.pose_ckpt)

            # check frame data already exists
            if self.points_layer is None:
                point_properties = {"confidence": [0], "ind": [0], "node": [0]}
                self.points_layer = self.viewer.add_points(
                    np.zeros((1, 3)),
                    properties=point_properties,
                    size=self.im_subset.data.shape[2] / 100,
                )

            points = []

            point_properties = self.points_layer.properties.copy()

            analysed_frames = np.unique(self.points_layer.data[:, 0])
            nframes = self.im_subset.data.shape[0]
            for frame in range(nframes):
                exists = False
                # for point in self.points_layer.data.tolist():  # its a list
                #    if point[0] == self.frame:
                #        print("point exists")
                if np.isin(analysed_frames, frame):
                    exists = True

                if exists == False:
                    person_results = []

                    if self.detection_layer is not None:
                        for nshape, shape in enumerate(
                            self.detection_layer.data
                        ):  # its a list
                            if shape[0, 0] == frame:
                                print(shape)
                                L = shape[0, 2]  # xmin
                                T = shape[0, 1]  # ymin
                                R = shape[2, 2]  # xmax
                                B = shape[2, 1]  # ymax

                                bbox_data = {
                                    "bbox": (L, T, R, B),
                                    "track_id": self.detection_layer.properties[
                                        "id"
                                    ][
                                        nshape
                                    ],
                                }

                                person_results.append(bbox_data)
                                im = self.im_subset.data[frame]

                    if len(person_results) > 0:
                        pose_results, _ = inference_top_down_pose_model(
                            self.model,
                            im,
                            person_results=person_results,
                            format="xyxy",
                        )

                    for ind in range(len(pose_results)):
                        keypoints = pose_results[ind]["keypoints"]
                        for ncoord in range(keypoints.shape[0]):
                            x, y, ci = keypoints[ncoord]

                            points.append((frame, y, x))
                            point_properties["confidence"] = np.append(
                                point_properties["confidence"], ci
                            )
                            point_properties["ind"] = np.append(
                                point_properties["ind"],
                                pose_results[ind]["track_id"],
                            )
                            point_properties["node"] = np.append(
                                point_properties["node"], ncoord
                            )

                    ## add part for if no bounding boxes

            # print(point_properties)
            # print(points)
            self.points_layer.data = np.concatenate(
                (self.points_layer.data, np.array(points))
            )

            # self.points_layer.properties[
            #    "confidence"
            # ] = point_properties["confidence"]
            # self.points_layer.properties["ind"] = point_properties[
            #    "ind"
            # ]
            # self.points_layer.properties["node"] = point_properties[
            #    "node"
            # ]
            self.points_layer.properties = point_properties

            df = pd.DataFrame(self.points_layer.properties)
            df2 = pd.DataFrame(self.points_layer.data)
            point_data = pd.concat([df, df2], axis=1)
            point_data.drop(0, axis=0, inplace=True)
            point_data.columns = ["ci", "ind", "node", "frame", "y", "x"]

            print(point_data.head())

            for ind in point_data.ind.unique():
                coord_data = {"x": None, "y": None, "ci": None}

                subset = point_data[point_data.ind == ind]

                for datum in ["x", "y", "ci"]:
                    empty = np.empty((self.n_nodes, nframes))
                    empty[:] = np.nan
                    subset_pivot = subset.pivot(
                        columns="frame", values=datum, index="node"
                    )

                    empty_df = pd.DataFrame(empty)
                    empty_df.loc[:, subset_pivot.columns] = subset_pivot

                    coord_data[datum] = empty_df

                self.coords_data[ind] = coord_data  # {
                # "x": subset.x,
                # "y": subset.y,
                # "z": subset.frame,
                # "ci": subset.ci,
                # }
                print(self.coords_data[ind])

            # print(self.points_layer.properties)

            # check for bounding boxes
            # call inference_top_down_mode(self.model, im)

    def predict_object_detection(self):
        self.initialise_params()

        if self.detection_backbone == "YOLOv5":
            self.model = torch.hub.load(
                "ultralytics/yolov5", "yolov5s", pretrained=True
            )

        elif self.detection_backbone == "YOLOv8":
            from ultralytics import YOLO

            # Load a model
            self.model = YOLO("yolov8m.pt")  # load an official model

        if self.accelerator == "gpu":
            self.device = torch.device("cuda")
        elif self.accelerator == "cpu":
            self.device = torch.device("cpu")

        self.model.to(self.device)

        if self.detection_layer == None:
            labels = ["0"]
            properties = {
                "label": labels,
            }

            # text_params = {
            #    "text": "label: {label}",
            #    "size": 12,
            #    "color": "green",
            #    "anchor": "upper_left",
            #    "translation": [-3, 0],
            #    }

            self.detection_layer = self.viewer.add_shapes(
                np.zeros((1, 4, 3)),
                shape_type="rectangle",
                edge_width=self.im_subset.data.shape[2] / 200,
                edge_color="#55ff00",
                face_color="transparent",
                visible=True,
                properties=properties,
                # text = text_params,
            )

        labels = self.detection_layer.properties["label"].tolist()
        shape_data = self.detection_layer.data
        ids = [0]
        colors = sns.color_palette("tab20", as_cmap=True)
        edge_colors = ["#55ff00"]

        print(f"shape data shape is {len(shape_data)}")
        print(f"labels are {labels}")
        # loop throuh frames
        if self.detection_backbone == "YOLOv5":
            for frame in range(self.im_subset.data.shape[0]):
                # assert frame is readable - some go pro ones seem corrupted for some reason

                exists = False

                try:
                    self.im_subset.data[frame]
                except:
                    exists = True

                # check frame data already exists

                for shape in self.detection_layer.data:  # its a list
                    if shape[0, 0] == frame:
                        print("bbox already exists")
                        exists = True
                        break

                if exists == False:
                    results = self.model(self.im_subset.data[frame])
                    result_df = results.pandas().xyxy[0]
                    print(result_df)
                    result_df = self.remove_overlapping_bboxes(result_df)
                    print(result_df)
                    for row in result_df.index:
                        labels.append(result_df.loc[row, "name"])

                        x_min = result_df.loc[row, "xmin"]
                        x_max = result_df.loc[row, "xmax"]
                        y_min = result_df.loc[row, "ymin"]
                        y_max = result_df.loc[row, "ymax"]

                        new_shape = np.array(
                            [
                                [frame, y_min, x_min],
                                [frame, y_min, x_max],
                                [frame, y_max, x_max],
                                [frame, y_max, x_min],
                            ]
                        )

                        shape_data.append(new_shape)
            print(labels)
            self.detection_layer.data = shape_data
            self.detection_layer.properties = {"label": labels, "id": ids}
            # map ids to boxes
            self.get_individual_ids()

        elif self.detection_backbone == "YOLOv8":
            h = self.im.shape[1]
            print(h)
            results = self.model.track(
                source=self.video_file,
                imgsz=h - (h % 32),
                tracker=os.path.join(self.decoder_data_dir, "botsort.yaml"),
                stream=True,
            )
            for frame, result in enumerate(results):
                names = result.names
                boxes = result.boxes
                for box in boxes:
                    # if box.conf > 0.1:
                    label = names[int(box.cls.cpu().numpy())]

                    print(frame)
                    if box.id is not None:
                        id = int(box.id.cpu().numpy()[0])
                        ids.append(id)
                        labels.append(label)
                        print(box.xyxy.cpu().numpy())
                        (
                            x_min,
                            y_min,
                            x_max,
                            y_max,
                        ) = box.xyxy.cpu().numpy()[0]
                        new_shape = np.array(
                            [
                                [frame, y_min, x_min],
                                [frame, y_min, x_max],
                                [frame, y_max, x_max],
                                [frame, y_max, x_min],
                            ]
                        )

                        shape_data.append(new_shape)

                    # shape_data = np.array(shape_data)

            print(labels)
            self.detection_layer.data = shape_data

            assert len(labels) == len(ids)
            self.detection_layer.properties = {"label": labels, "id": ids}
            new_edge_colors = [colors.colors[idx] for idx in ids]
            self.detection_layer.edge_color = new_edge_colors

        self.populate_chkpt_dropdown()
        self.label_menu.choices = self.choices

    def remove_overlapping_bboxes(self, result_df, thresh=0.999):
        # check no overlap
        corr_df = result_df[["xmin", "xmax", "ymin", "ymax"]].T.corr()
        corr_df[corr_df == 1] = 0  # set diagonal to 0

        # look for similar bboxes
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        corr_df = corr_df[pd.DataFrame(mask)]
        rows, cols = np.where(corr_df.to_numpy() > thresh)

        for nrow, ncol in zip(rows, cols):
            # find and keep most confident

            # check if nrow in index or already removed
            # check if ncol in index or already removed
            if nrow not in result_df.index:
                pass  # don't need to drop anything

            elif ncol not in result_df.index:
                pass  # don't need to drop anything

            else:
                row_conf = result_df.loc[nrow, "confidence"]
                col_conf = result_df.loc[ncol, "confidence"]
                print(f"overlapping {row_conf} {col_conf} ")

                max_idx = np.argmax((row_conf, col_conf))

                if max_idx == 0:
                    # drop ncow
                    result_df.drop(ncol, inplace=True)
                elif max_idx == 1:
                    result_df.drop(nrow, inplace=True)

        return result_df

    def get_bbox_center(self, box):
        xmin, xmax, ymin, ymax = box[0, -1], box[1, -1], box[0, 1], box[2, 1]
        center = (np.median([xmin, xmax]), np.median([ymin, ymax]))
        return center, xmin, xmax, ymin, ymax

    def create_graph_from_bboxes(self, bboxes, euclidean=False):
        pos = []
        num_nodes = len(bboxes)
        feats = []

        # normalise coords to width and height
        w = self.im.shape[2]
        h = self.im.shape[1]

        center_fov = (w / 2, h / 2)

        for box in bboxes:
            center, xmin, xmax, ymin, ymax = self.get_bbox_center(box)

            center = (np.array(center) - center_fov) / np.array([w, h])

            xmin, ymin = (np.array([xmin, ymin]) - center_fov) / np.array(
                [w, h]
            )

            xmax, ymax = (np.array([xmax, ymax]) - center_fov) / np.array(
                [w, h]
            )

            pos.append(center)
            feats.append([center[0], center[1], xmin, xmax, ymin, ymax])

        # reshape pos
        pos1 = np.array(pos).reshape(num_nodes, 2, 1)
        pos1 = np.tile(pos1, num_nodes)
        posT = pos1.T
        diff = posT - pos1

        if euclidean:
            # create euclidean adjacency matrix
            A = np.sqrt((diff[:, 0] ** 2) + (diff[:, 0] ** 2))

        else:
            A = np.ones((num_nodes, num_nodes))

        A = torch.from_numpy(A)
        torch.diagonal(A)[:] = 0
        G = nx.from_numpy_array(A.numpy())
        for npos, position in enumerate(pos):
            G.nodes[npos]["x"] = position[0]
            G.nodes[npos]["y"] = position[1]
        n = torch.tensor([num_nodes])
        # F = torch.tensor(feats)
        F = torch.from_numpy(np.array(feats))

        # center_fov
        # F = (F - F.mean(axis=0)) / F.std(axis=0)

        return G, A, n, F

    def get_bboxes_by_frame(self, frame, detection_layer):
        box_idx = np.where(np.array(detection_layer.data)[:, 0, 0] == frame)[0]
        boxes = np.array(detection_layer.data)[box_idx]
        return box_idx, boxes

    def graph_matching(self, frame1, frame2):
        idx1, bboxes1 = self.get_bboxes_by_frame(frame1, self.detection_layer)
        idx2, bboxes2 = self.get_bboxes_by_frame(frame2, self.detection_layer)

        G1, A1, n1, F1 = self.create_graph_from_bboxes(bboxes1)
        G2, A2, n2, F2 = self.create_graph_from_bboxes(bboxes2)

        conn1, edge1 = pygm.utils.dense_to_sparse(A1)
        conn2, edge2 = pygm.utils.dense_to_sparse(A2)

        import functools

        gaussian_aff = functools.partial(
            pygm.utils.gaussian_aff_fn, sigma=0.001
        )  # set affinity function
        K = pygm.utils.build_aff_mat(
            F1,
            edge1,
            conn1,
            F2,
            edge2,
            conn2,
            n1,
            None,
            n2,
            None,
            edge_aff_fn=gaussian_aff,
        )

        X = pygm.rrwm(K, n1, n2)  # pygm.sm(K, n1, n2)
        match = pygm.hungarian(X)
        return match

        # To DO - modeify track id to represent new id
        # To DO- import networkx and pym -add to package dependencies
        # Create vector layer for connections using distance matrix and origin points

    def get_individual_ids(self):
        # get modal frame
        from scipy.stats import mode

        pygm.BACKEND = "pytorch"

        nframes = self.im.shape[0]

        max_ind_frame, max_inds = mode(
            np.array(self.detection_layer.data)[:, 0, 0].flatten()
        )
        # create forwards and backwards frames from that
        max_ind_frame = max_ind_frame[0]
        max_inds = max_inds[0]
        forwards_frames = np.arange(max_ind_frame, nframes - 1)
        backwards_frames = np.flip(np.arange(1, max_ind_frame + 1))
        # create frame id dataframe to track which object is associated with id
        id_df = pd.DataFrame(
            [], columns=np.arange(max_inds), index=np.arange(nframes)
        )
        id_df.loc[max_ind_frame] = pd.Series(np.arange(max_inds))

        # loop through frames and match scene graphs
        id_df = self.match_frames(forwards_frames, id_df, self.detection_layer)
        id_df = self.match_frames(
            backwards_frames, id_df, self.detection_layer
        )

        props = self.detection_layer.properties
        ids = np.zeros(len(props["label"]))

        # map to detection layer properties
        for frame in np.arange(0, nframes):
            idx, bboxes = self.get_bboxes_by_frame(frame, self.detection_layer)
            rev_box_map = {
                int(v): k
                for k, v in id_df.loc[frame]
                .dropna()
                .sort_values()
                .to_dict()
                .items()
            }
            if len(rev_box_map) > 0:
                accounted_idx = idx[list(rev_box_map.keys())]
                ids[accounted_idx] = list(rev_box_map.values())

        props["id"] = ids
        self.detection_layer.properties = props
        return id_df

    def match_frames(self, frames, df, detection_layer):
        for nframe, frame in enumerate(frames):
            try:
                match = self.graph_matching(frame, frames[nframe + 1])

                # any unmatched column ids - new ind - try match against prev mean
                unmatched = match.numpy().sum(axis=0) == 0
                if unmatched.any():
                    match2 = self.attempt_mean_match(
                        frames[nframe + 1], match, df
                    )
                    if match2 is None:
                        pass
                    else:
                        match = match2

                previous, next_ = np.where(match == 1)
                # get reverse dict of object label from previous frame
                rev_dict = {v: k for k, v in df.loc[frame].to_dict().items()}
                # map previous object label to true id
                true_id = pd.Series(previous).map(rev_dict, na_action="ignore")
                for n_id, id in enumerate(true_id):
                    if np.isnan(id):
                        print("null_true_id")

                    else:
                        df.loc[frames[nframe + 1], id] = next_[n_id]

                unmatched = match.numpy().sum(axis=0) == 0
                if unmatched.any():  # if col id still unmatched add new ind
                    df.loc[frames[nframe + 1], df.columns[-1] + 1] = np.where(
                        unmatched
                    )[0][0]

            except:
                print(f"no individuals present frame {frame}")

        return df

    def attempt_mean_match(self, frame_to_match, match, id_df):
        nearest = []
        for frame in np.flip(np.arange(frame_to_match - 20, frame_to_match)):
            if frame > 0:
                idx, bboxes = self.get_bboxes_by_frame(
                    frame, self.detection_layer
                )
                if len(idx) == match.shape[1]:
                    nearest.append(frame)

        adjacencies = []
        features = []

        for frame in nearest:
            idx, bboxes = self.get_bboxes_by_frame(frame, self.detection_layer)
            G, A, n, F = self.create_graph_from_bboxes(bboxes)
            # correct F and A
            new_order = id_df.loc[frame].dropna().to_numpy().astype("int64")
            print(new_order)  # possible weirdness here with new_order

            if new_order.shape[0] == match.shape[1]:
                new_A = self.reorder_adjacency(A.numpy(), new_order)
                new_F = F.numpy()[new_order]
                adjacencies.append(new_A)
                features.append(new_F)

        mean_F = np.array(features).mean(axis=0)
        mean_A = np.array(adjacencies).mean(axis=0)

        try:
            match = self.graph_match_against_mean(
                frame_to_match, mean_A, mean_F
            )
        except:
            print(f"mean match failed - frame to match is {frame_to_match}")
            match = None
        return match

    def reorder_adjacency(self, A, new_order):
        new_A = np.zeros(A.shape)
        for nrow, row in enumerate(new_order):
            for ncol, col in enumerate(new_order):
                new_A[nrow, ncol] = A[row, col]
        return new_A

    def graph_match_against_mean(self, frame, mean_A, mean_F):
        mean_A = torch.from_numpy(mean_A)
        mean_F = torch.from_numpy(mean_F)
        idx1, bboxes1 = self.get_bboxes_by_frame(frame, self.detection_layer)
        G2, A2, n2, F2 = self.create_graph_from_bboxes(bboxes1)

        n1 = n2
        conn1, edge1 = pygm.utils.dense_to_sparse(mean_A)
        conn2, edge2 = pygm.utils.dense_to_sparse(A2)

        import functools

        gaussian_aff = functools.partial(
            pygm.utils.gaussian_aff_fn, sigma=0.001
        )  # set affinity function
        K = pygm.utils.build_aff_mat(
            mean_F,
            edge1,
            conn1,
            F2,
            edge2,
            conn2,
            n1,
            None,
            n2,
            None,
            edge_aff_fn=gaussian_aff,
        )

        X = pygm.rrwm(K, n1, n2)  # pygm.sm(K, n1, n2)
        match = pygm.hungarian(X)
        return match

    def preprocess_bouts(self):
        # arrange in N, C, T, V format
        self.zebdata = ZebData()
        points = self.egocentric[:, :, 1:]
        points = np.swapaxes(points, 0, 2)
        ci_array = self.ci.to_numpy()
        ci_array = ci_array.reshape((*ci_array.shape, 1))
        cis = np.swapaxes(ci_array, 0, 2)

        # N, C, T, V, M - don't ignore confidence interval but give option of removing
        N = len(self.behaviours)
        C = self.config_data["train_cfg"]["num_channels"]
        T2 = self.config_data["data_cfg"]["T2"]
        denominator = self.config_data["data_cfg"]["denominator"]
        T_method = self.config_data["data_cfg"]["T"]
        fps = self.config_data["data_cfg"]["fps"]

        if T_method == "window":
            T = 2 * int(fps / denominator)

        elif type(T_method) == "int":
            T = T_method  # these methods assume behaviours last the same amount of time -which is a big assumption

        elif T_method == "None":
            T = 43
        V = points.shape[2]
        M = 1

        bouts = np.zeros((N, C, T, V, M))
        padded_bouts = np.zeros((N, C, T2, V, M))

        # loop through movement windows when behaviour occuring
        for n, (bhv_start, bhv_end) in enumerate(self.behaviours):
            # focus on window of size tsne window around peak of movement
            bhv_mid = bhv_start + ((bhv_end - bhv_start) / 2)
            new_start = int(
                bhv_mid - int(T / 2)
            )  # might be worth refining self behaviours from here
            new_end = int(bhv_mid + int(T / 2))
            new_end = (
                T - (new_end - new_start)
            ) + new_end  # this adds any difference if not exactly T in length

            if T_method == "window":
                # if window method then refine the behaviours to the new start and end so that when analyse is called points and video align correctly
                self.behaviours[n] = (new_start, new_end)

            bhv = points[:, new_start:new_end]
            ci = cis[:, new_start:new_end]
            ci = ci.reshape((*ci.shape, 1))

            # switch to x, y from y, x
            bhv_rs = bhv.copy()
            bhv_rs[1] = bhv[0]
            bhv_rs[0] = bhv[1]
            bhv = bhv_rs

            # reflect y to convert to cartesian friendly coordinate system - is this needed if coordinates are egocentric?
            bhv = bhv.reshape((*bhv.shape, 1))  # reshape to N, C, T, V, M
            y_mirror = np.array([[1, 0], [0, -1]])

            for frame in range(bhv.shape[1]):
                bhv[:2, frame] = (bhv[:2, frame].T @ y_mirror).T

            # align function takes shape N, C, T, V, M
            bhv_align = self.zebdata.align(bhv)

            bouts[n, :2] = bhv_align
            bouts[n, 2] = ci[0]

            padded_bouts[n] = self.zebdata.pad(bouts[n], T2)

        self.zebdata.data = padded_bouts
        self.zebdata.labels = np.zeros(padded_bouts.shape[0])

    def predict_behaviours(self):
        data_cfg, graph_cfg, hparams = self.initialise_params()

        data_loader = DataLoader(
            self.zebdata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
        )

        # load model check point,
        log_folder = os.path.join(self.decoder_data_dir, "lightning_logs")
        self.chkpt = os.path.join(
            log_folder, self.chkpt_dropdown.value
        )  # spinbox

        try: # old pytorch lightning
            model = st_gcn_aaai18_pylightning_3block.ST_GCN_18(
                in_channels=self.numChannels,
                num_workers=self.num_workers,
                num_class=self.numlabels,
                graph_cfg=graph_cfg,
                data_cfg=data_cfg,
                hparams=hparams,
            ).load_from_checkpoint(
                self.chkpt,
                in_channels=self.numChannels,
                num_workers=self.num_workers,
                num_class=self.numlabels,
                graph_cfg=graph_cfg,
                data_cfg=data_cfg,
                hparams=hparams,
            )
        except:
            model = st_gcn_aaai18_pylightning_3block.ST_GCN_18.load_from_checkpoint(self.chkpt, data_cfg=data_cfg)

        # create trainer object,
        ## optimise trainer to just predict as currently its preparing data thinking its training
        # predict
        trainer = Trainer(devices=self.devices, accelerator=self.accelerator)
        predictions = trainer.predict(
            model, data_loader
        )  # returns a list of the processed batches
        self.predictions = torch.concat(predictions, dim=0)
        print(self.predictions)

    def update_classification_data_with_predictions(self):
        label_dict = self.config_data["data_cfg"]["classification_dict"]

        # add predictions to classification data
        # if self.model_dropdown.value == "ZebLR":
        #    label_dict = {0 : "forward",
        #                  1: "left",
        #                  2: "right"}
        #
        preds = torch.argmax(self.predictions, dim=1).numpy()

        print(type(preds))
        print(f"predictions is {preds}")

        self.b_labels = pd.Series(preds).map(label_dict).to_numpy()
        print(self.b_labels)

        self.choices = tuple(label_dict.values())
        self.label_menu.choices = self.choices

        # maybe loop and create new classification_data
        if len(self.classification_data[self.ind].keys()) == len(
            self.behaviours
        ):
            # loop classification data and just change label
            for nb, (b, b_data) in enumerate(
                self.classification_data[self.ind].items()
            ):
                b_data["classification"] = self.b_labels[nb]
        else:
            # invoke behaviour changed loop - very slow
            for b in range(len(self.behaviours)):
                self.behaviour_changed(b + 1)
            self.behaviour_changed(0)

    def convert_classification_files(self, train_files):
        # use folder and convert classification files
        train_bouts = []
        train_labels = []

        classification_files = [
            tb.open_file(file, mode="r") for file in train_files
        ]
        for file in classification_files:
            classification_data = self.read_classification_h5(file)
            C = self.config_data["train_cfg"][
                "num_channels"
            ]  # 3 # publicise these
            V = self.config_data["data_cfg"]["V"]  #    19
            M = 1
            fps = self.config_data["data_cfg"]["fps"]  # 330.
            denominator = self.config_data["data_cfg"]["denominator"]  # 8

            T_method = self.config_data["data_cfg"]["T"]

            if T_method == "window":
                T = 2 * int(fps / denominator)

            elif type(T_method) == "int":
                T = T_method  # these methods assume behaviours last the same amount of time -which is a big assumption

            elif T_method == "None":
                # ragged nest, T should be max length and everything padded to that
                T = None  # T2 should be specified to ensure same length

            center = self.config_data["data_cfg"]["center"]
            T2 = self.config_data["data_cfg"]["T2"]
            align = self.config_data["data_cfg"]["align"]
            all_bouts, all_labels = self.classification_data_to_bouts(
                classification_data,
                C,
                T,
                V,
                M,
                center=center,
                T2=T2,
                align=align,
            )
            train_bouts.append(all_bouts)
            train_labels.append(all_labels)

        train_bouts = np.concatenate(train_bouts)
        train_labels = np.array(
            [item for sublist in train_labels for item in sublist]
        )

        return train_bouts, train_labels

    def prepare_data(self):
        # Prepare and save data
        # take one datafolder and
        all_files = [
            os.path.join(self.decoder_data_dir, file)
            for file in os.listdir(self.decoder_data_dir)
            if "classification.h5" in file
        ]
        nfiles = len(all_files)

        train_proportion = int(0.85 * nfiles)

        self.train_files = all_files[:train_proportion]
        self.test_files = all_files[train_proportion:]

        self.train_data, self.train_labels = self.convert_classification_files(
            self.train_files
        )
        self.test_data, self.test_labels = self.convert_classification_files(
            self.test_files
        )

        # drop labels to ignore
        labels_to_ignore = self.config_data["data_cfg"]["labels_to_ignore"]
        print(f"Labels to ignore are {labels_to_ignore}")
        good_train_idx = np.where(
            ~np.isin(self.train_labels, labels_to_ignore)
        )[0]
        good_test_idx = np.where(~np.isin(self.test_labels, labels_to_ignore))[
            0
        ]
        print(
            f"Subset of train labels ignoring some labels {self.train_labels[good_train_idx[:10]]}"
        )
        # check if any labels to ignore are still in
        print(f"Unique labels are {np.unique(self.train_labels)}")

        self.train_data, self.train_labels = (
            self.train_data[good_train_idx],
            self.train_labels[good_train_idx],
        )
        self.test_data, self.test_labels = (
            self.test_data[good_test_idx],
            self.test_labels[good_test_idx],
        )

        self.class_dict = self.config_data["data_cfg"]["classification_dict"]
        self.label_dict = {v: k for k, v in self.class_dict.items()}
        # label_dict = {k:v for v, k in enumerate(np.unique(self.train_labels))}
        print(f"Label dict is {self.label_dict}")

        self.train_labels = (
            pd.Series(self.train_labels).map(self.label_dict).to_numpy()
        )
        self.test_labels = (
            pd.Series(self.test_labels).map(self.label_dict).to_numpy()
        )

        # check if any labels to ignore are still in
        print(f"Unique labels are {np.unique(self.train_labels)}")
        print(
            f"Label_counts are {pd.Series(self.train_labels).value_counts()}"
        )

        print(f"Training Data Shape is {self.train_data.shape}")
        print(f"Test Data Shape is {self.test_data.shape}")

        # np.save(os.path.join(self.decoder_data_dir, "label_dict.npy"), self.label_dict)
        self.augmentation = self.config_data["data_cfg"]["augmentation"]
        # if self.augmentation is not False:
        #    zebdata = ZebData()
        #    zebdata.data = self.train_data
        #    zebdata.labels = self.train_labels
        #    zebdata.ideal_sample_no = self.augmentation
        #    zebdata.dynamic_augmentation()

        #    self.train_data = zebdata.data
        #    self.train_labels = zebdata.labels

        #    print("Augmented Training Data Shape is {}".format(self.train_data.shape))

        np.save(
            os.path.join(self.decoder_data_dir, "Zebtrain.npy"),
            self.train_data,
        )
        np.save(
            os.path.join(self.decoder_data_dir, "Zebtrain_labels.npy"),
            self.train_labels,
        )
        np.save(
            os.path.join(self.decoder_data_dir, "Zebtest.npy"), self.test_data
        )
        np.save(
            os.path.join(self.decoder_data_dir, "Zebtest_labels.npy"),
            self.test_labels,
        )

        print(f"Data Prepared and Save at {self.decoder_data_dir}")

    def live_decode(self, event):
        print(f"Live checkbox is {self.live_checkbox.value}")
        if self.live_checkbox.value:
            data_cfg, graph_cfg, hparams = self.initialise_params()

            if self.model_dropdown.value == "Detection":
                if self.detection_backbone == "YOLOv5":
                    self.model = torch.hub.load(
                        "ultralytics/yolov5", "yolov5s", pretrained=True
                    )

                elif self.detection_backbone == "YOLOv8":
                    from ultralytics import YOLO

                    # Load a model
                    self.model = YOLO("yolov8m.pt")

                if self.accelerator == "gpu":
                    self.device = torch.device("cuda")
                elif self.accelerator == "cpu":
                    self.device = torch.device("cpu")

                self.model.to(self.device)

            elif self.model_dropdown.value == "PoseEstimation":
                if self.pose_config is not None:
                    self.model = init_pose_model(
                        self.pose_config, self.pose_ckpt
                    )

            elif self.model_dropdown.value == "BehaviourDecode":
                log_folder = os.path.join(
                    self.decoder_data_dir, "lightning_logs"
                )
                self.chkpt = os.path.join(
                    log_folder, self.chkpt_dropdown.value
                )  # spinbox
                if self.backbone == "ST-GCN":
                    self.model = st_gcn_aaai18_pylightning_3block.ST_GCN_18(
                        in_channels=self.numChannels,
                        num_class=self.numlabels,
                        num_workers=self.num_workers,
                        graph_cfg=graph_cfg,
                        data_cfg=data_cfg,
                        hparams=hparams,
                    ).load_from_checkpoint(
                        self.chkpt,
                        in_channels=self.numChannels,
                        num_workers=self.num_workers,
                        num_class=self.numlabels,
                        graph_cfg=graph_cfg,
                        data_cfg=data_cfg,
                        hparams=hparams,
                    )

                self.model.freeze()

                if self.accelerator == "gpu":
                    self.device = torch.device("cuda")
                elif self.accelerator == "cpu":
                    self.device = torch.device("cpu")

                self.model.to(self.device)

                self.ethogram_im = np.zeros(
                    (self.numlabels, self.dlc_data.shape[0])
                )
                # self.viewer1d.clear_canvas()

                self.ethogram = self.viewer1d.add_image(
                    self.ethogram_im,
                    blending="opaque",
                    colormap="inferno",
                    visible=True,
                )
                print(f"Model succesfully loaded onto device {self.device}")

                # elif self.backbone == "C3D":
                #   model = c3d.C3D(num_class =self.numlabels,
                #                    num_channels = self.numChannels,
                #                    data_cfg = data_cfg,
                #                    hparams= hparams,
                #                    num_workers = self.num_workers
                #                    )

    def train(self):
        # self.decoder_data_dir = self.decoder_dir_picker.value
        # Load prepare data
        if os.path.exists(os.path.join(self.decoder_data_dir, "Zebtrain.npy")):
            print("Data Prepared")

        else:
            print("Preparing Data")
            self.prepare_data()

        # train

        data_cfg, graph_cfg, hparams = self.initialise_params()

        print(data_cfg, graph_cfg, hparams)

        # create trainer object,
        ## optimise trainer to just predict as currently its preparing data thinking its training
        # predict
        for n in range(self.numRuns):  # does ths reuse model in current state?
            if self.backbone == "ST-GCN":
                model = st_gcn_aaai18_pylightning_3block.ST_GCN_18(
                    in_channels=self.numChannels,
                    num_workers=self.num_workers,
                    num_class=self.numlabels,
                    graph_cfg=graph_cfg,
                    data_cfg=data_cfg,
                    hparams=hparams,
                )
            elif self.backbone == "C3D":
                model = c3d.C3D(
                    num_class=self.numlabels,
                    num_channels=self.numChannels,
                    data_cfg=data_cfg,
                    hparams=hparams,
                    num_workers=self.num_workers,
                )

            #for param in model.parameters():
            #    if param.requires_grad:
                    #print(f"param {param} requires grad")

            print(f"trial is {n}")

            TTLogger = TensorBoardLogger(save_dir=self.decoder_data_dir)
            
            print("Early stop metric is {} and mode is {} and patience is {}".format(self.early_stop_metric, self.early_stop_mode, self.patience))
            early_stop = EarlyStopping(
                monitor=self.early_stop_metric, mode=self.early_stop_mode, patience=self.patience
            )
            swa = StochasticWeightAveraging(swa_lrs=1e-2)

            log_folder = os.path.join(self.decoder_data_dir, "lightning_logs")
            if os.path.exists(log_folder):
                if len(os.listdir(log_folder)) > 0:
                    version_folders = [
                        version_folder
                        for version_folder in os.listdir(log_folder)
                        if "version" in version_folder
                    ]
                    latest_version_number = max(
                        [
                            int(version_folder.split("_")[-1])
                            for version_folder in version_folders
                        ]
                    )  # this is not working quite right not selectin latest folder
                    print(f"latest version folder is {latest_version_number}")
                    new_version_number = latest_version_number + 1
                    new_version_folder = os.path.join(
                        log_folder, f"version_{new_version_number}"
                    )
                    print(new_version_folder)

                else:
                    new_version_folder = os.path.join(log_folder, "version_0")

            else:
                new_version_folder = os.path.join(log_folder, "version_0")

            print(f"new version folder is {new_version_folder}")

            checkpoint_callback = ModelCheckpoint(
                monitor=self.early_stop_metric,
                dirpath=new_version_folder,
                filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}--{auprc:.2f}--{elapsed_time:.2f}",
                save_top_k=1,  # save the best model
                mode=self.early_stop_mode,
                every_n_epochs=1,
            )

            ## Run this 4 times and select best model - fine tune that
            try:
                 # old pytorch lightning
                trainer = Trainer(
                    logger=TTLogger,
                    devices=1,
                    accelerator=self.accelerator,
                    max_epochs=100,
                    callbacks=[early_stop, checkpoint_callback],
                    auto_lr_find=True,
                )  # , stochastic_weight_avg=True) - this is a callback in latest lightning-, swa -swa messes up auto lr

                trainer.tune(model)

                ### DEBUG - overfit
                # trainer = Trainer(devices =1, accelerator = "gpu", overfit_batches=0.01)
            except:
                # new pytorch lightning
                from pytorch_lightning.tuner import Tuner
                trainer = Trainer(
                    logger=TTLogger,
                    devices=1,
                    accelerator=self.accelerator,
                    max_epochs=100,
                    callbacks=[early_stop, checkpoint_callback],
                )  # , stochastic_weight_avg=True) - this is a callback in latest lightning-, swa -swa messes up auto lr
                tuner = Tuner(trainer)
                tuner = Tuner(trainer)
                tuner.lr_find(model)

            trainer.fit(model)

            print(
                f"Finished Training - best model is {checkpoint_callback.best_model_path}"
            )

            # load new checkpoints
            self.populate_chkpt_dropdown()
            # Add finetune - freeze model-replace last layer and train

    def initialise_params(self):
        self.numlabels = self.config_data["data_cfg"]["numLabels"]
        self.devices = self.config_data["train_cfg"]["devices"]
        self.auto_lr = self.config_data["train_cfg"]["auto_lr"]
        self.accelerator = self.config_data["train_cfg"]["accelerator"]
        self.graph_layout = self.config_data["train_cfg"]["graph_layout"]
        self.dropout = self.config_data["train_cfg"]["dropout"]
        self.numChannels = self.config_data["train_cfg"]["num_channels"]
        self.num_workers = self.config_data["train_cfg"]["num_workers"]
        
        # create dataloader from preprocess swims
        self.batch_size = self.batch_size_spinbox.value  # spinbox
        # self.num_workers = self.num_workers_spinbox.value # spinbox
        self.lr = self.lr_spinbox.value  # spinbox
        # self.dropout = self.dropout_spinbox.value # spinbox
        try:
            self.regress = self.config_data["train_cfg"]["regress"]
        except:
            print("not regression")
            self.regress = False
        try:
            self.head_node = self.config_data["train_cfg"]["head_node"]
        except:
            self.head_node = 0
        try:
            self.backbone = self.config_data["train_cfg"]["backbone"]

        except:
            "print no backbone- defaulting to STGCN"
            self.backbone = "ST-GCN"

        try:
            self.transform = self.config_data["train_cfg"]["transform"]
            print(f"transform is {self.transform}")
            if self.transform == "None":
                self.transform = None
        except:
            self.transform = None

        try:
            self.labels_to_ignore = self.config_data["data_cfg"][
                "labels_to_ignore"
            ]
            if self.labels_to_ignore == "None":
                self.labels._to_ignore = None
        except:
            self.labels_to_ignore = None

        try:
            self.augmentation = self.config_data["data_cfg"]["augmentation"]
            if (self.augmentation == "None") | (self.augmentation == False):
                print("No augmentation")
                self.augment = False
                #self.ideal_sample_no = None

            else:
                print(f"Augmenting data {self.augmentation}")
                self.augment = True
                #self.ideal_sample_no = self.augmentation
        except:
            self.augment = False
            #self.ideal_sample_no = None
        try:
            self.ideal_sample_no = self.config_data["data_cfg"]["ideal_sample_no"]
        except:
            self.ideal_sample_no = None

        try:
            self.dataset = self.config_data["dataset"]

        except:
            self.dataset = None

        try:
            self.detection_backbone = self.config_data["train_cfg"][
                "detection_backbone"
            ]
        except:
            self.detection_backbone = None

        try:
            self.pose_config = os.path.join(
                self.decoder_data_dir,
                self.config_data["train_cfg"]["pose_config"],
            )
            self.pose_ckpt = os.path.join(
                self.decoder_data_dir,
                self.config_data["train_cfg"]["pose_ckpt"],
            )
        except:
            self.pose_config = None
            self.pose_ckpt = None

        try:
            self.calc_class_weights = self.config_data["data_cfg"][
                "calc_class_weights"
            ]

        except:
            self.calc_class_weights = False
        try:
            self.class_dict = self.config_data["data_cfg"][
                "classification_dict"
            ]
            self.label_dict = {v: k for k, v in self.class_dict.items()}
            # label_dict = {k:v for v, k in enumerate(np.unique(self.train_labels))}
            print(f"Label dict is {self.label_dict}")
        except:
            self.class_dict = None
            self.label_dict = None

        try:
            self.T2 = self.config_data["data_cfg"]["T2"]
        except:
            self.T2 = None

        try: 
            self.softmax = self.config_data["train_cfg"]["softmax"]
        except:
            self.softmax = False

        try: 
            self.weighted_random_sampler = self.config_data["data_cfg"]["weighted_random_sampler"]
        except:
            self.weighted_random_sampler = None

        try:
            self.early_stop_metric = self.config_data["train_cfg"]["early_stop_metric"]
            self.early_stop_mode = self.config_data["train_cfg"]["early_stop_mode"]
            self.patience = self.config_data["train_cfg"]["patience"]
        except:
            self.early_stop_metric = "val_loss"
            self.early_stop_mode = "min"
            self.patience = 5
        try:
            self.numRuns = self.config_data["train_cfg"]["numRuns"]
        except:
            self.numRuns = 4

        try:
            self.freeze = self.config_data["train_cfg"]["freeze"]
        except:
            self.freeze = False

        try:
            self.binary = self.config_data["train_cfg"]["binary"]
            self.binary_class = self.config_data["train_cfg"]["binary_class"]
        except:
            self.binary = False
            self.binary_class = None

        try:
            self.preprocess_frame = self.config_data["train_cfg"]["preprocess_frame"]
            self.window_size = self.config_data["train_cfg"]["window_size"]
        except:
            self.preprocess_frame = False
            self.window_size = None


        # assign model parameters
        PATH_DATASETS = self.decoder_data_dir
        # self.numlabels = self.num_labels_spinbox.value # spinbox
        # self.numChannels = self.num_channels_spinbox.value # X, Y and CI - spinbox

        data_cfg = {
            "data_dir": PATH_DATASETS,
            "augment": self.augment,
            "ideal_sample_no": self.ideal_sample_no,
            "shift": False,
            "transform": self.transform,
            "labels_to_ignore": self.labels_to_ignore,
            "label_dict": self.label_dict,
            "calc_class_weights": self.calc_class_weights,
            "regress": self.regress,
            "T2": self.T2,
            "head" : self.head_node,
            "softmax": self.softmax,
            "weighted_random_sampler": self.weighted_random_sampler,
            "binary": self.binary,
            "binary_class": self.binary_class,
        }

        graph_cfg = {"layout": self.graph_layout, "center": self.center_node}

        hparams = HyperParams(self.batch_size, self.lr, self.dropout)

        return (data_cfg, graph_cfg, hparams)

    def finetune(self):
        ### Fine tune strategies - 1) Freeze and modify last layer, 2) Train on worse perfoming classes

        # load best checkpoint

        # freeze model
        # model.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        # create model
        # load checkpoint
        # set all layers to grad = False
        # change configure optimised in st -gcn backbone
        # train
        data_cfg, graph_cfg, hparams = self.initialise_params()

        # orig_num_labels = (
        #    len(data_cfg["labels_to_ignore"]) + self.numlabels
        # )  # this is for specific example
        orig_num_labels = self.numlabels

        log_folder = os.path.join(self.decoder_data_dir, "lightning_logs")
        self.chkpt = os.path.join(
            log_folder, self.chkpt_dropdown.value
        )  # spinbox

        for n in range(4):  # does ths reuse model in current state?
            try: 
                model = st_gcn_aaai18_pylightning_3block.ST_GCN_18(
                in_channels=self.numChannels,
                num_workers=self.num_workers,
                num_class=orig_num_labels,  # self.numlabels,
                graph_cfg=graph_cfg,
                data_cfg=data_cfg,
                hparams=hparams,
                ).load_from_checkpoint(
                    self.chkpt,
                    in_channels=self.numChannels,
                    num_workers=self.num_workers,
                    num_class=orig_num_labels,  # self.numlabels,
                    graph_cfg=graph_cfg,
                    data_cfg=data_cfg,
                    hparams=hparams,
                )
            except:
                try:
                    model = st_gcn_aaai18_pylightning_3block.ST_GCN_18.load_from_checkpoint(self.chkpt, data_cfg=data_cfg)
                except:
                    model = st_gcn_aaai18_pylightning_3block.ST_GCN_18.load_from_checkpoint(self.chkpt, num_class = self.numlabels-1) # use specifc case -delete

                if model.num_classes != orig_num_labels:
                    print("num classes don't match - assume transfer learning")
                    model.num_classes = orig_num_labels
                    
                #    #model.setup("fit")
            if self.freeze:
                # freeze model layers
                for param in model.parameters():
                    param.requires_grad = False

            #print(model.parameters)

            # add new model.fcn
            model.fcn = nn.Conv2d(256, self.numlabels, kernel_size=1)

            #for param in model.parameters():
            #    if param.requires_grad:
            #        print(f"param {param} requires grad")

            print(f"trial is {n}")

            TTLogger = TensorBoardLogger(save_dir=self.decoder_data_dir)
            print("Early stop metric is {} and mode is {}".format(self.early_stop_metric, self.early_stop_mode))
            early_stop = EarlyStopping(
                monitor=self.early_stop_metric, mode=self.early_stop_mode, patience=self.patience
            )
            swa = StochasticWeightAveraging(swa_lrs=1e-2)

            if os.path.exists(log_folder):
                if len(os.listdir(log_folder)) > 0:
                    version_folders = [
                        version_folder
                        for version_folder in os.listdir(log_folder)
                        if "version" in version_folder
                    ]
                    latest_version_number = max(
                        [
                            int(version_folder.split("_")[-1])
                            for version_folder in version_folders
                        ]
                    )  # this is not working quite right not selectin latest folder
                    print(f"latest version folder is {latest_version_number}")
                    new_version_number = latest_version_number + 1
                    new_version_folder = os.path.join(
                        log_folder, f"version_{new_version_number}"
                    )
                    print(new_version_folder)

                else:
                    new_version_folder = os.path.join(log_folder, "version_0")

            else:
                new_version_folder = os.path.join(log_folder, "version_0")

            print(f"new version folder is {new_version_folder}")

            checkpoint_callback = ModelCheckpoint(
                monitor=self.early_stop_metric,
                dirpath=new_version_folder,
                filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}--{auprc:.2f}--{elapsed_time:.2f}",
                save_top_k=1,  # save the best model
                mode=self.early_stop_mode,
                every_n_epochs=1,
            )
            try:
                 # old pytorch lightning
                trainer = Trainer(
                    logger=TTLogger,
                    devices=1,
                    accelerator=self.accelerator,
                    max_epochs=100,
                    callbacks=[early_stop, checkpoint_callback],
                    auto_lr_find=True,
                )  # , stochastic_weight_avg=True) - this is a callback in latest lightning-, swa -swa messes up auto lr

                trainer.tune(model)

                ### DEBUG - overfit
                # trainer = Trainer(devices =1, accelerator = "gpu", overfit_batches=0.01)
            except:
                # new pytorch lightning
                from pytorch_lightning.tuner import Tuner
                trainer = Trainer(
                    logger=TTLogger,
                    devices=1,
                    accelerator=self.accelerator,
                    max_epochs=100,
                    callbacks=[early_stop, checkpoint_callback],
                )  # , stochastic_weight_avg=True) - this is a callback in latest lightning-, swa -swa messes up auto lr
                tuner = Tuner(trainer)
                tuner = Tuner(trainer)
                tuner.lr_find(model)

            trainer.fit(model)

            print(
                f"Finished Finetuning - best model is {checkpoint_callback.best_model_path}"
            )

    def test(self):
        data_cfg, graph_cfg, hparams = self.initialise_params()

        log_folder = os.path.join(self.decoder_data_dir, "lightning_logs")
        self.chkpt = os.path.join(
            log_folder, self.chkpt_dropdown.value
        )  # spinbox
        try:
            model = st_gcn_aaai18_pylightning_3block.ST_GCN_18(
                in_channels=self.numChannels,
                num_class=self.numlabels,
                num_workers=self.num_workers,
                graph_cfg=graph_cfg,
                data_cfg=data_cfg,
                hparams=hparams,
            ).load_from_checkpoint(
                self.chkpt,
                in_channels=self.numChannels,
                num_workers=self.num_workers,
                num_class=self.numlabels,
                graph_cfg=graph_cfg,
                data_cfg=data_cfg,
                hparams=hparams,
            )
        except:
            model = st_gcn_aaai18_pylightning_3block.ST_GCN_18.load_from_checkpoint(self.chkpt, 
                                                                                    in_channels=self.numChannels,
                                                                                    num_workers=self.num_workers,
                                                                                    num_class=self.numlabels,
                                                                                    graph_cfg=graph_cfg,
                                                                                    data_cfg=data_cfg,
                                                                                    hparams=hparams,)
            
            
        model.freeze()

        #TTLogger = TensorBoardLogger(save_dir=self.decoder_data_dir)
        #early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        #swa = StochasticWeightAveraging(swa_lrs=1e-2)

        if os.path.exists(log_folder):
            if len(os.listdir(log_folder)) > 0:
                version_folders = [
                    version_folder
                    for version_folder in os.listdir(log_folder)
                    if "version" in version_folder
                ]
                latest_version_number = max(
                    [
                        int(version_folder.split("_")[-1])
                        for version_folder in version_folders
                    ]
                )  # this is not working quite right not selectin latest folder
                print(f"latest version folder is {latest_version_number}")
                new_version_number = latest_version_number + 1
                new_version_folder = os.path.join(
                    log_folder, f"version_{new_version_number}"
                )
                print(new_version_folder)

            else:
                new_version_folder = os.path.join(log_folder, "version_0")

        else:
            new_version_folder = os.path.join(log_folder, "version_0")

        print(f"new version folder is {new_version_folder}")

        trainer = Trainer(
            #logger=TTLogger,
            devices=self.devices,
            accelerator=self.accelerator,
            max_epochs=100,
            #profiler="advanced"
            #callbacks=[early_stop],
            #auto_lr_find=True,
        )  # , stochastic_weight_avg=True) - this is a callback in latest lightning-, swa -swa messes up auto lr

        trainer.test(model)
        predictions = trainer.predict(model, model.test_dataloader())

        predictions = torch.concat(predictions, dim=0)

        np.save(
            os.path.join(os.path.dirname(self.chkpt), "predictions.npy"),
            predictions.numpy(),
        )

        print("Finished Testing")

        print("Benchmarking model performance")
        self.benchmark_model_performance(model)
        #self.benchmark_model_flops(model)

    def read_classification_h5(self, file):
        classification_data = {}
        for group in file.root.__getattr__("_v_groups"):
            ind = file.root[group]
            behaviour_dict = {}
            arrays = {}

            for array in file.list_nodes(ind, classname="Array"):
                arrays[int(array.name)] = array
            tables = []

            for table in file.list_nodes(ind, classname="Table"):
                tables.append(table)

            behaviours = []
            classifications = []
            starts = []
            stops = []
            cis = []
            for row in tables[0].iterrows():
                behaviours.append(row["number"])
                classifications.append(row["classification"])
                starts.append(row["start"])
                stops.append(row["stop"])

            for behaviour, classification, start, stop in zip(
                behaviours, classifications, starts, stops
            ):
                class_dict = {
                    "classification": classification.decode("utf-8"),
                    "coords": arrays[behaviour][:, :3],
                    "start": start,
                    "stop": stop,
                    "ci": arrays[behaviour][:, 3],
                }
                behaviour_dict[behaviour + 1] = class_dict

            classification_data[int(group)] = behaviour_dict
        file.close()
        return classification_data

    def classification_data_to_bouts(
        self, classification_data, C, T, V, M, center=None, T2=None, align=None
    ):
        print(type(classification_data))
        all_ind_bouts = []
        all_labels = []
        for ind in classification_data.keys():
            behaviour_dict = classification_data[ind]
            N = len(behaviour_dict.keys())

            if (T2 == "None") | (T2 is None):
                bout_data = np.zeros((N, C, T, V, M))
            else:
                bout_data = np.zeros((N, C, T2, V, M))
            bout_labels = []
            for bout_idx, bout in enumerate(behaviour_dict.keys()):
                # get coords
                coords = behaviour_dict[bout]["coords"]

                # reshape coords to V, T, (frame, Y, X)
                coords_reshaped = coords.reshape(V, -1, 3)

                print(coords_reshaped.shape)

                # get ci
                ci = behaviour_dict[bout]["ci"]
                ci_reshaped = ci.reshape(V, -1, 1)

                # subset behaviour from the middle out
                # focus on window of size tsne window around peak of movement

                if T == None:
                    # take behaviour as is
                    # pad to T2
                    print("T  is none")
                    coords_subset = coords_reshaped
                    ci_subset = ci_reshaped

                else:
                    mid_idx = int(coords_reshaped.shape[1] / 2)
                    new_start = int(mid_idx - int(T / 2))
                    new_end = int(mid_idx + int(T / 2))
                    new_end = (
                        T - (new_end - new_start)
                    ) + new_end  # this adds any difference if not exactly T in length

                    coords_subset = coords_reshaped[:, new_start:new_end]
                    print(new_start, new_end)
                    print(coords_subset.shape)
                    ci_subset = ci_reshaped[:, new_start:new_end]

                # reshape from V, T, C to C, T, V
                swapped_coords = np.swapaxes(coords_subset, 0, 2)
                new_bout = np.zeros(swapped_coords.shape)
                print(new_bout.shape)
                swapped_ci = np.swapaxes(ci_subset, 0, 2)

                new_bout[0] = swapped_coords[2]  # x
                new_bout[1] = swapped_coords[1]  # y
                new_bout[2] = swapped_ci[0]  # ci

                # reflect y to convert to cartesian friendly coordinate system
                new_bout = new_bout.reshape(
                    (*new_bout.shape, M)
                )  # reshape to N, C, T, V, M
                y_mirror = np.array([[1, 0], [0, -1]])

                for frame in range(new_bout.shape[1]):
                    new_bout[:2, frame] = (new_bout[:2, frame].T @ y_mirror).T

                # align, pad,

                zebdata = ZebData()

                if center != "None":
                    print(f"centering bout on center {center}")
                    new_bout = zebdata.center_all(new_bout, center)
                    # new_bout = centered_bout.copy()

                if align:
                    print("aligning bout")
                    new_bout = zebdata.align(new_bout)

                if T2 != "None":
                    print(
                        f"padding bout, original size was {new_bout.shape}, new T is {T2}"
                    )
                    new_bout = zebdata.pad(new_bout, T2)

                bout_data[bout_idx] = new_bout
                label = behaviour_dict[bout]["classification"]
                bout_labels.append(label)

            all_ind_bouts.append(bout_data)
            all_labels.append(bout_labels)

        all_ind_bouts = np.concatenate(all_ind_bouts)
        all_labels = np.array(all_labels).flatten()

        return all_ind_bouts, all_labels

    def view_data(self, view=False):
        if view:
            try:
                train_data = np.load(
                    os.path.join(self.decoder_data_dir, "Zebtrain.npy")
                )
                train_labels = np.load(
                    os.path.join(self.decoder_data_dir, "Zebtrain_labels.npy")
                )
                self.transform = self.config_data["train_cfg"]["transform"]
                self.batch_size = self.batch_size_spinbox.value
                self.zebdata = ZebData(transform=self.transform)
                self.zebdata.data = train_data
                self.zebdata.labels = train_labels

                self.spinbox.max = self.zebdata.labels.shape[0]
            except:
                print("No training data")

    def show_data(self, idx):
        print(self.zebdata[idx][0].shape)
        if self.transform == "heatmap":
            if self.im_subset != None:
                self.im_subset.data = np.max(
                    self.zebdata[idx][0].numpy(), axis=0
                )
            else:
                self.im_subset = self.viewer.add_image(
                    np.max(self.zebdata[idx][0].numpy(), axis=0),
                    name="Video Recording",
                )

    def benchmark_model_performance(self, model):
        # get inference speed of one sample
        # get time taken to process 1, 10, 100, 1000 behaviours

        model.batch_size = 1
        dataloader = model.test_dataloader()

        if self.accelerator == "gpu":
            device = torch.device("cuda")
        elif self.accelerator == "cpu":
            device = torch.device("cpu")

        model.to(device)

        #### Single sample

        dummy_data = next(iter(dataloader))[0].to(device)
        # print(dummy_data)

        starter, ender = torch.cuda.Event(
            enable_timing=True
        ), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))
        # GPU-WARM-UP
        for _ in range(10):
            _ = model(dummy_data)
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = model(dummy_data)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(f"Mean inference latency on one sample is {mean_syn} ms")

        ### A series of behaviours
        model.batch_size = 10

        durations = {}

        for n, nsamples in enumerate([1, 10, 100, 1000]):
            for trial in range(4):
                # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                repetitions = int(np.ceil(nsamples / model.batch_size))
                print(repetitions)
                dataloader = model.test_dataloader()
                with torch.no_grad():
                    # starter.record()
                    start = time.time()
                    for rep in range(repetitions):
                        dummy_data = next(iter(dataloader))[0].to(device)
                        _ = model(dummy_data)
                    # WAIT FOR GPU SYNC
                    # torch.cuda.synchronize()
                    # ender.record()
                    # curr_time = starter.elapsed_time(ender)
                    end = time.time()
                    curr_time = end - start
                    durations[str(nsamples), trial] = curr_time

        print(f"Durations are {durations}")

        np.save(
            os.path.join(self.decoder_data_dir, "inference_latencies.npy"),
            timings,
        )
        np.save(
            os.path.join(
                self.decoder_data_dir, "inference_durations_vs_datasetsize.npy"
            ),
            durations,
        )

    def benchmark_model_flops(self, model):
        from lightning.fabric.utilities.throughput import measure_flops
        model.batch_size = 16
        model.to(torch.device("meta"))
        x = next(iter(model.dataloader))[0].to(torch.device("meta"))

        model_fwd = lambda: model(x)
        fwd_flops = measure_flops(model, model_fwd)

       
        print(f"FLOPS for forward pass: {fwd_flops}")
        



    def save_ethogram(self):
        # instead of saving the bout info in classification data - save the ethogram instead

        df = pd.DataFrame()
        filename = str(self.video_file) + "_classification.csv"
        for ind in self.classification_data.keys():
            for behaviour in self.classification_data[ind].keys():
                label = self.classification_data[ind][behaviour][
                    "classification"
                ]
                start = self.classification_data[ind][behaviour]["start"]
                stop = self.classification_data[ind][behaviour]["stop"]
                df = pd.concat(
                    [df, pd.Series([ind, behaviour, start, stop, label])],
                    axis=1,
                )
        df = df.T
        df.columns = ["individual", "nbehaviour", "start", "stop", "behaviour"]
        # df["duration"] =

        df.to_csv(filename)

        # summarise data
        count_filename = filename = (
            str(self.video_file) + "_behaviour_counts.csv"
        )
        count_df = df.behaviour.value_counts()
        count_df.to_csv(count_filename)


class Behaviour(tb.IsDescription):
    number = tb.Int32Col()
    classification = tb.StringCol(16)
    n_nodes = tb.Int32Col()
    start = tb.Int32Col()
    stop = tb.Int32Col()
