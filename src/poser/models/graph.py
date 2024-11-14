r"""Copyright [2018] [@misc{mmskeleton2019,
  author =       {Sijie Yan, Yuanjun Xiong, Jingbo Wang, Dahua Lin},
  title =        {MMSkeleton},
  howpublished = {\url{https://github.com/open-mmlab/mmskeleton}},
  year =         {2019}
}]
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modifications made by [Pierce Mullen] on [11-11-24]"""

import numpy as np

### Original code from Sijie Yan, Yuanjun Xiong, Jingbo Wang, Dahua Lin (2019)
class Graph:
    """The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(
        self,
        layout="zebrafish",
        strategy="spatial",
        max_hop=1,
        dilation=1,
        center=0,
    ):
        self.max_hop = max_hop
        self.dilation = dilation
        self.layout = layout
        self.center = center
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop
        )
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == "openpose":
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 3),
                (3, 2),
                (7, 6),
                (6, 5),
                (13, 12),
                (12, 11),
                (10, 9),
                (9, 8),
                (11, 5),
                (8, 2),
                (5, 1),
                (2, 1),
                (0, 1),
                (15, 0),
                (14, 0),
                (17, 15),
                (16, 14),
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "ntu-rgb+d":
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1, 2),
                (2, 21),
                (3, 21),
                (4, 3),
                (5, 21),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 21),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
                (22, 23),
                (23, 8),
                (24, 25),
                (25, 12),
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == "ntu_edge":
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1, 2),
                (3, 2),
                (4, 3),
                (5, 2),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 2),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
                (21, 22),
                (22, 8),
                (23, 24),
                (24, 12),
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == "coco":
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [16, 14],
                [14, 12],
                [17, 15],
                [15, 13],
                [12, 13],
                [6, 12],
                [7, 13],
                [6, 7],
                [8, 6],
                [9, 7],
                [10, 8],
                [11, 9],
                [2, 3],
                [2, 1],
                [3, 1],
                [4, 2],
                [5, 3],
                [4, 6],
                [5, 7],
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0

        # Code modified by Pierce Mullen
        elif layout == "zebrafish":
            # print(layout)
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 5

        elif layout == "drosophila":
            # print(layout)
            self.num_node = 12
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 6],
                [6, 7],
                [7, 8],
                [7, 1],
                [7, 5],
                [8, 9],
                [9, 10],
                [10, 11],
                [11, 2],
                [11, 4],
                [2, 3],
                [4, 3],
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 8

        elif layout == "zebrafishlarvae":
            # print(layout)
            self.num_node = 19
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [0, 5],
                [1, 2],
                [1, 4],
                [2, 3],
                [4, 3],
                [5, 6],
                [5, 8],
                [8, 7],
                [6, 7],
                [7, 9],
                [3, 9],
                [0, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [12, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [16, 17],
                [17, 18],
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 13

        elif layout == "zeb60fps":
            # print(layout)
            self.num_node = 23
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [8, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [12, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [16, 17],
                [17, 18],
                [18, 19],
                [19, 20],
                [20, 21],
                [21, 22],
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0

        elif layout == "mouse1":
            self.num_node = 13
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 3],
                [0, 1],
                [0, 4],
                [1, 2],
                [2, 3],
                [2, 4],
                [3, 6],
                [3, 5],
                [4, 7],
                [4, 5],
                [5, 7],
                [5, 6],
                [6, 8],
                [7, 9],
                [5, 8],
                [5, 9],
                [5, 10],
                [9, 10],
                [8, 10],
                [10, 11],
                [11, 12],
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 5

        elif layout == "mouse2":
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                [0, 5],
                [1, 5],
                [2, 5],
                [3, 5],
                [4, 5],
                [5, 8],
                [5, 6],
                [5, 9],
                [6, 7],
                [7, 8],
                [7, 9],
                [8, 11],
                [8, 10],
                [9, 12],
                [9, 10],
                [10, 12],
                [10, 11],
                [11, 13],
                [12, 14],
                [10, 13],
                [10, 14],
                [10, 15],
                [14, 15],
                [13, 15],
                [15, 16],
                [16, 17],
            ]

            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 10

        elif type(layout) == list:
            print(layout)
            self.num_node = np.unique(np.array(layout)).shape[0]
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = layout
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link

        else:
            # print(layout)
            raise ValueError("Do Not Exist This Layout.")


    ### Original code from Sijie Yan, Yuanjun Xiong, Jingbo Wang, Dahua Lin (2019)
    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == "uniform":
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[
                    self.hop_dis == hop
                ]
            self.A = A
        elif strategy == "spatial":
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if (
                                self.hop_dis[j, self.center]
                                == self.hop_dis[i, self.center]
                            ):
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif (
                                self.hop_dis[j, self.center]
                                > self.hop_dis[i, self.center]
                            ):
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")

### Original code from Sijie Yan, Yuanjun Xiong, Jingbo Wang, Dahua Lin (2019)
def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
