import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import HTML, display
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from ._loader import ZebData


def plotting_palette():
    """Defines a custom colorblind seaborn palette.

    Returns:
        palette : Seaborn palette"""

    custom = [
        (0, 0, 0),
        (255, 0, 102),
        (16, 127, 128),
        (64, 0, 127),
        (107, 102, 255),
        (102, 204, 254),
    ]
    custom = [tuple(np.array(list(rgb)) / 255) for rgb in custom]
    custom2 = [
        sns.light_palette(color, n_colors=3, reverse=True) for color in custom
    ]
    custom3 = np.array(custom2).reshape([18, 3]).tolist()
    custom4 = custom3[0::3] + custom3[1::3] + custom3[2::3]
    palette = sns.color_palette(custom4)
    # sns.palplot(palette)
    return palette


def rotate_transform(behaviour, numAngles):
    """Rotates poses returning a set number of rotated poses.

    # N, C, T, V, M"""

    rotated = np.zeros((numAngles, *behaviour.shape))

    for angle_no in range(numAngles):
        # random angle between -30 and + 30
        angle = (np.random.random(1) * 60) - 30
        angle = np.radians(angle[0])

        # rotation matrix to use to transform coordinate space
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, s], [-s, c]])  # clockwise

        # rotate all time points in behaviour by multiplying rotation matrix with behaviour X, Y
        transformed = np.dot(R, behaviour[:2].reshape(2, -1)).reshape(
            behaviour[0:2, :].shape
        )
        rotated[angle_no] = behaviour.copy()
        rotated[angle_no, :2] = transformed

    return rotated


def jitter_transform(behaviour, numJitter):
    """Adds noise to poses returning a set number of rotated poses.

    # N, C, T, V, M"""

    jittered = np.zeros((numJitter, *behaviour.shape))

    for jitter_no in range(numJitter):
        # random jitter between -5 and +5 pixels
        jitter = (np.random.random(behaviour[:2].shape) * 4) - 2

        jittered[jitter_no] = behaviour.copy()
        jittered[jitter_no, :2] = behaviour[:2] + jitter

    return jittered


def scale_transform(behaviour, numScales):
    """Randomly scales poses"""

    scaled = np.zeros((numScales, *behaviour.shape))

    for scale_no in range(numScales):
        # create random scales between 0 and 3
        scale = np.random.random(1) * 3

        scaled[scale_no] = behaviour.copy()
        scaled[scale_no, :2] = behaviour[:2] * scale

    return scaled


def shear_transform(behaviour, numShears):
    sheared = np.zeros((numShears, *behaviour.shape))

    for shear_no in range(numShears):
        # create random scales between -1.5 and 1.5
        shear_x = (np.random.random(1) * 2) - 1
        shear_y = np.random.random(1) * 1

        shear_matrix = np.array([[1, shear_x[0]], [shear_y[0], 1]])

        transformed = np.dot(
            shear_matrix, behaviour[:2].reshape(2, -1)
        ).reshape(behaviour[0:2, :].shape)
        sheared[shear_no] = behaviour.copy()
        sheared[shear_no, :2] = transformed

    return sheared


def create_graph(behaviour):
    G = nx.Graph()
    G.add_node(0, x=behaviour[0, 0][0], y=behaviour[1, 0][0])
    G.add_node(1, x=behaviour[0, 1][0], y=behaviour[1, 1][0])
    G.add_node(2, x=behaviour[0, 2][0], y=behaviour[1, 2][0])

    G.add_edges_from([[0, 1], [1, 2], [2, 0]])
    return G


def extract_bouts(
    viewer,
    h5_files,
    fps=330.0,
    window_denominator=8,
    activity_threshold=2,
    confidence_threshold=0.8,
):
    # viewer = Viewer([], n_nodes = 19, center_node = 13)
    zebdata = ZebData()

    viewer.fps = fps
    viewer.tsne_window = 2 * int(
        viewer.fps / window_denominator
    )  # this was changed to 2x

    all_bouts = []
    bout_map = pd.DataFrame()

    for h5_file in h5_files:
        viewer.h5_file = h5_file
        viewer.read_coords(viewer.h5_file)
        for key in viewer.coords_data.keys():
            # get points
            viewer.x = viewer.coords_data[key]["x"]
            viewer.y = viewer.coords_data[key]["y"]
            viewer.ci = viewer.coords_data[key]["ci"]
            viewer.get_points()

            # calculate movement
            viewer.calulate_orthogonal_variance(
                amd_threshold=activity_threshold,
                confidence_threshold=confidence_threshold,
            )

            # arrange in N, C, T, V format
            points = viewer.egocentric[:, :, 1:]
            points = np.swapaxes(points, 0, 2)
            ci_array = viewer.ci.to_numpy()
            ci_array = ci_array.reshape((*ci_array.shape, 1))
            cis = np.swapaxes(ci_array, 0, 2)

            # N, C, T, V, M - don't ignore confidence interval but give option of removing
            N = len(viewer.behaviours)
            C = 3
            T = viewer.tsne_window
            V = points.shape[2]
            M = 1

            bouts = np.zeros((N, C, T, V, M))
            bouts.shape
            all_starts = []
            all_ends = []

            # loop through movement windows when behaviour occuring
            for n, (bhv_start, bhv_end) in enumerate(viewer.behaviours):
                # focus on window of size tsne window around peak of movement
                bhv_mid = bhv_start + ((bhv_end - bhv_start) / 2)
                new_start = int(bhv_mid - int(T / 2))
                new_end = int(bhv_mid + int(T / 2))
                new_end = (T - (new_end - new_start)) + new_end
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
                bhv_align = zebdata.align(bhv)

                bouts[n, :2] = bhv_align
                bouts[n, 2] = ci[0]
                all_starts.append(new_start)
                all_ends.append(new_end)

            all_bouts.append(bouts)
            # create a data frame to store metadata
            # number, file, start, end
            file_bout_index = pd.DataFrame(
                [np.arange(N), [h5_file] * N, all_starts, all_ends]
            ).T
            # file_bout_index.reset_index(inplace = True)
            bout_map = pd.concat([bout_map, file_bout_index])

    all_bout_array = np.concatenate(all_bouts)
    all_bout_array.shape
    # bout_map.drop("index", axis=1, inplace = True)
    bout_map.columns = ["nbout", "file", "start", "end"]
    bout_map.reset_index(drop=True, inplace=True)

    return all_bout_array, bout_map


def extract_bouts_from_filedict(
    viewer,
    file_dict,
    fps=330.0,
    window_denominator=8,
    activity_threshold=2,
    confidence_threshold=0.8,
):
    # viewer = Viewer([], n_nodes = 19, center_node = 13)
    zebdata = ZebData()

    viewer.fps = fps
    viewer.tsne_window = 2 * int(viewer.fps / window_denominator)

    all_bouts = []
    bout_map = pd.DataFrame()

    for fish, (h5_file, log_file, protocol_dict) in file_dict.items():
        viewer.h5_file = h5_file
        viewer.read_coords(viewer.h5_file)
        for key in viewer.coords_data.keys():
            # get points
            viewer.x = viewer.coords_data[key]["x"]
            viewer.y = viewer.coords_data[key]["y"]
            viewer.ci = viewer.coords_data[key]["ci"]
            viewer.get_points()

            # calculate movement
            viewer.calulate_orthogonal_variance(
                amd_threshold=activity_threshold,
                confidence_threshold=confidence_threshold,
            )

            # arrange in N, C, T, V format
            points = viewer.egocentric[:, :, 1:]
            points = np.swapaxes(points, 0, 2)
            ci_array = viewer.ci.to_numpy()
            ci_array = ci_array.reshape((*ci_array.shape, 1))
            cis = np.swapaxes(ci_array, 0, 2)

            # N, C, T, V, M - don't ignore confidence interval but give option of removing
            N = len(viewer.behaviours)
            C = 3
            T = viewer.tsne_window
            V = points.shape[2]
            M = 1

            bouts = np.zeros((N, C, T, V, M))
            bouts.shape
            all_starts = []
            all_ends = []

            # loop through movement windows when behaviour occuring
            for n, (bhv_start, bhv_end) in enumerate(viewer.behaviours):
                # focus on window of size tsne window around peak of movement
                bhv_mid = bhv_start + ((bhv_end - bhv_start) / 2)
                new_start = int(bhv_mid - int(T / 2))
                new_end = int(bhv_mid + int(T / 2))
                new_end = (
                    T - (new_end - new_start)
                ) + new_end  # what is the point in this again
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
                bhv_align = zebdata.align(bhv)

                bouts[n, :2] = bhv_align
                bouts[n, 2] = ci[0]
                all_starts.append(new_start)
                all_ends.append(new_end)

            all_bouts.append(bouts)
            # create a data frame to store metadata
            # number, file, start, end
            file_bout_index = pd.DataFrame(
                [np.arange(N), [h5_file] * N, all_starts, all_ends]
            ).T

            # map protocol into swim bout dictionary
            file_bout_index["protocol"] = [np.nan] * file_bout_index.shape[0]
            for protocol, (start, end) in protocol_dict.items():
                start_filter = file_bout_index.loc[:, 2] > start
                end_filter = file_bout_index.loc[:, 2] <= end
                file_bout_index.loc[
                    start_filter & end_filter, "protocol"
                ] = protocol
            # file_bout_index.reset_index(inplace = True)
            bout_map = pd.concat([bout_map, file_bout_index])

    all_bout_array = np.concatenate(all_bouts)
    all_bout_array.shape
    # bout_map.drop("index", axis=1, inplace = True)
    bout_map.columns = ["nbout", "file", "start", "end", "protocol"]
    bout_map.reset_index(drop=True, inplace=True)

    return all_bout_array, bout_map


def read_classification_h5(file):
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
    return classification_data


def classification_data_to_bouts(
    classification_data, C, T, V, M, fps, denominator, center=None
):
    all_ind_bouts = []
    all_labels = []
    for ind in classification_data.keys():
        behaviour_dict = classification_data[ind]
        N = len(behaviour_dict.keys())
        bout_data = np.zeros((N, C, T, V, M))
        bout_labels = []
        for bout_idx, bout in enumerate(behaviour_dict.keys()):
            # get coords
            coords = behaviour_dict[bout]["coords"]

            # reshape coords to V, T, (frame, Y, X)
            coords_reshaped = coords.reshape(V, -1, 3)

            # get ci
            ci = behaviour_dict[bout]["ci"]
            ci_reshaped = ci.reshape(V, -1, 1)

            # subset behaviour from the middle out
            mid_idx = int(coords_reshaped.shape[1] / 2)
            new_start = mid_idx - int(fps / denominator)
            new_end = mid_idx + int(fps / denominator)
            coords_subset = coords_reshaped[:, new_start:new_end]
            ci_subset = ci_reshaped[:, new_start:new_end]

            # reshape from V, T, C to C, T, V
            swapped_coords = np.swapaxes(coords_subset, 0, 2)
            new_bout = np.zeros(swapped_coords.shape)
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

            if center != None:
                centered_bout = zebdata.center_all(new_bout, center)
                aligned_bout = zebdata.align(centered_bout)
                padded_bout = zebdata.pad(aligned_bout, T)
                new_bout = padded_bout

            bout_data[bout_idx] = new_bout
            label = behaviour_dict[bout]["classification"]
            bout_labels.append(label)

        all_ind_bouts.append(bout_data)
        all_labels.append(bout_labels)

    all_ind_bouts = np.concatenate(all_ind_bouts)
    all_labels = np.array(all_labels).flatten()

    return all_ind_bouts, all_labels


def convert_dlc_to_ctvm(dlc_file):
    if dlc_file.endswith(".h5"):
        dlc_data = pd.read_hdf(dlc_file)
    elif dlc_file.endswith(".csv"):
        dlc_data = pd.read_csv(dlc_file, header=[0, 1, 2], index_col=0)
    data_t = dlc_data.transpose()
    data_t["individuals"] = ["individual1"] * data_t.shape[0]
    data_t = (
        data_t.reset_index()
        .set_index(["scorer", "individuals", "bodyparts", "coords"])
        .reset_index()
    )
    ctvms = []
    for individual in data_t.individuals.unique():
        indv = data_t[data_t.individuals == individual].copy()
    
        bodyparts = data_t.bodyparts.unique()

        x= data_t[data_t.coords == "x"].loc[:, 0:].to_numpy()
        x = x.reshape(1, len(bodyparts), -1, 1)    

        y= data_t[data_t.coords == "y"].loc[:, 0:].to_numpy()
        y = y.reshape(1, len(bodyparts), -1, 1)  

        ci = data_t[data_t.coords == "likelihood"].loc[:, 0:].to_numpy()
        ci = ci.reshape(1, len(bodyparts), -1, 1)

        CVTM = np.concatenate([x, y, ci], axis=0) # CVTM
        CTVM = np.swapaxes(CVTM, 1, 2) # CTVM
        ctvms.append(CTVM)

    # concatenate along M axis
    CTVM = np.concatenate(ctvms, axis = -1)

    return CTVM

# function for converting napari points layer to CTVM
# function for converting yolo data to CTVM


# Define animation class for viewing poses


class Animation:
    def __init__(
        self, dataset, skeleton, label_dict=None, shuffle=True, normalise=False, batch_size = 8
    ):
        super().__init__()
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
        self.dataset = dataset
        self.skeleton = skeleton
        self.label_dict = label_dict
        self.normalise = normalise
        self.N, self.C, self.T, self.V, self.M = self.dataset.data.shape

    # def setup(self):
    #    self.scat = self.ax.scatter(self.bhv[:, 0], cmap="jet", edgecolor="k") # use a rainbow colormap here
    #    self.ax.set_ylim(top = 300, bottom = -300)
    #    self.ax.set_xlim(left = -300, right = 300)
    #    return self.scat,

    def pose_to_graph(self, skeleton, n_nodes, frame):
        self.G = nx.Graph()
        self.G.add_nodes_from(np.arange(n_nodes))
        self.G.add_edges_from(skeleton)
        array = self.bhv[:2, frame]
        self.pos = {k: tuple(array[:, k]) for k in range(array.shape[1])}

    def plot_graph(self):
        M = self.G.number_of_edges()
        N = self.G.number_of_nodes()
        edge_colors = range(2, M + 2)
        node_colors = range(2, N + 2)
        edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
        cmap = plt.cm.plasma
        self.ax.cla()
        self.nodes = nx.draw_networkx_nodes(
            self.G,
            pos=self.pos,
            node_color=node_colors,
            cmap=cmap,
            node_size=20,
            ax=self.ax,
        )
        self.edges = nx.draw_networkx_edges(
            self.G,
            pos=self.pos,
            edge_color=edge_colors,
            edge_cmap=cmap,
            width=2,
            ax=self.ax,
        )
        # ax = plt.gca()

        self.ax.set_ylim(bottom=-self.max, top=self.max)
        self.ax.set_xlim(left=-self.max, right=self.max)
        self.ax.axis("off")
        return (
            self.nodes,
            self.edges,
        )

    def setup(self):
        if self.normalise:
            self.max = 5
            self.behaviours_x_mean = self.behaviours[:, 0].mean()
            self.behaviours_y_mean = self.behaviours[:, 1].mean()
            self.behaviours_x_std = self.behaviours[:, 0].std()
            self.behaviours_y_std = self.behaviours[:, 1].std()
            self.behaviours[:, 0] = (
                self.behaviours[:, 0] - self.behaviours_x_mean
            ) / self.behaviours_x_std
            self.behaviours[:, 1] = (
                self.behaviours[:, 1] - self.behaviours_y_mean
            ) / self.behaviours_y_std

        else:
            self.max = np.nanmax(np.abs(self.behaviours[:, :2]))
        for bhv in range(self.behaviours.shape[0]):
            self.ax = self.axes[bhv]
            self.bhv = self.behaviours[bhv].reshape((self.C, self.T, -1))

            # if self.normalise:
            #    self.normalise_bhv()

            self.pose_to_graph(self.skeleton, self.V, 0)
            self.plot_graph()
            return (self.nodes,)

    def normalise_bhv(self):
        self.bhv[0] = (
            self.bhv[0] - self.behaviours_x_mean
        ) / self.behaviours_x_std
        self.bhv[1] = (
            self.bhv[1] - self.behaviours_y_mean
        ) / self.behaviours_y_std

    def update(self, frame):
        # loop through create graph function
        for bhv in range(self.behaviours.shape[0]):
            self.ax = self.axes[bhv]
            self.bhv = self.behaviours[bhv].reshape((self.C, self.T, -1))
            # if self.normalise:
            #    self.normalise_bhv()

            skeleton = self.skeleton

            self.pose_to_graph(self.skeleton, self.V, frame)
            self.plot_graph()
            for previous_frame in range(frame):
                self.ax.plot(
                    self.bhv[0, previous_frame],
                    self.bhv[1, previous_frame],
                    color="gray",
                    linewidth=0.3,
                    alpha=0.5,
                )

            if self.label_dict:
                self.ax.set_title(self.label_dict[int(self.labels[bhv])])
            else:
                self.ax.set_title(str(self.labels[bhv]))
            self.ax.tick_params(
                left=True, bottom=True, labelleft=True, labelbottom=True
            )
            sns.despine()

        return (self.nodes,)

    def plot_kde(self, frame):
        self.x, self.y = np.swapaxes(
            self.bhv_type[:, :2, frame], 0, 1
        ).reshape(2, -1)
        self.ax.cla()
        sns.kdeplot(
            x=self.x,
            y=self.y,
            fill=True,
            gridsize=500,
            levels=15,
            thresh=0.1,
            ax=self.ax,
        )
        self.ax.set_ylim(bottom=-100, top=100)
        self.ax.set_xlim(left=-100, right=100)

    def setup_kde(self):
        for n, label in enumerate(np.unique(self.dataset.labels)):
            self.ax = self.axes[n]
            self.bhv_type = self.dataset.data[self.dataset.labels == label]

            self.plot_kde(0)
            return (self.ax.get_children()[0],)

    def update_kde(self, frame):
        for n, label in enumerate(np.unique(self.dataset.labels)):
            self.ax = self.axes[n]
            self.bhv_type = self.dataset.data[self.dataset.labels == label]

            self.plot_kde(frame)
            return (self.ax.get_children()[0],)

    def animate(self, repeat=False):
        self.behaviours, self.labels = next(iter(self.dataloader))
        self.fig, self.axes = plt.subplots(
            ncols=self.dataloader.batch_size, figsize=(20, 10)
        )
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=self.behaviours.shape[2],
            blit=True,
            repeat=repeat,
            interval=1000 / 10,
            init_func=self.setup,
        )
        # display(HTML(self.ani.to_jshtml()))

    def animate_kde(self):
        self.fig, self.axes = plt.subplots(
            ncols=np.unique(self.dataset.labels).shape[0], figsize=(40, 10)
        )
        self.ani = FuncAnimation(
            self.fig,
            self.update_kde,
            frames=20,
            blit=True,
            repeat=True,
            interval=1000 / 10,
            init_func=self.setup_kde,
        )
        display(HTML(self.ani.to_jshtml()))
