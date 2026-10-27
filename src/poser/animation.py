"""Animate pose bouts for notebook use.

Kept out of core because it needs matplotlib.pyplot, which STYLEGUIDE 1.2
excludes from the core layer.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import seaborn as sns
import torch
from IPython.display import HTML, display
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


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
