import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from pore.models import ConvTemporalGraphical, Graph
from pore.preprocessing import PreProcessing
from pore.recognition import ZebData
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Lambda


def zero(x):
    return 0


def iden(x):
    return x


class ST_GCN_18(LightningModule):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(
        self,
        in_channels,
        num_class,
        graph_cfg,
        data_cfg,
        hparams,
        batch_size=0,
        num_workers=0,
        edge_importance_weighting=True,  # try changing to false
        data_bn=True,
        **kwargs
    ):
        super().__init__()
        # self.hparams.update(hparams)
        self.num_workers = num_workers

        self.data_dir = data_cfg["data_dir"]
        self.augment = data_cfg["augment"]
        self.ideal_sample_no = data_cfg["ideal_sample_no"]
        self.shift = data_cfg["shift"]

        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size

        self.num_classes = num_class
        self.save_hyperparameters("hparams")

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False
        )
        self.register_buffer("A", A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = (
            nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else iden
        )

        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}
        self.st_gcn_networks = nn.ModuleList(
            (
                st_gcn_block(
                    in_channels, 64, kernel_size, 1, residual=False, **kwargs0
                ),
                st_gcn_block(
                    64, 64, kernel_size, 1, **kwargs
                ),  # remove to trim
                st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(
                    64, 64, kernel_size, 1, **kwargs
                ),  # remove to trim
                st_gcn_block(64, 128, kernel_size, 2, **kwargs),
                st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(
                    128, 128, kernel_size, 1, **kwargs
                ),  # remove to trim
                st_gcn_block(128, 256, kernel_size, 2, **kwargs),
                st_gcn_block(
                    256, 256, kernel_size, 1, **kwargs
                ),  # remove to trim
                st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            )
        )

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [
                    nn.Parameter(torch.ones(self.A.size()))
                    for i in self.st_gcn_networks
                ]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Make sure dataloaders are on cuda
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        preds = torch.argmax(output, dim=1)
        acc = accuracy(preds, y)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            logger=True,
            on_epoch=True,
        )
        self.log("train_acc", acc, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        preds = torch.argmax(output, dim=1)
        acc = accuracy(preds, y)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            logger=True,
            on_epoch=True,
        )
        self.log("val_acc", acc, prog_bar=True, logger=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_train_start(self):
        # self.hparams = {"lr": self.learning_rate,
        #                "batch_size": self.batch_size}
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/learning_rate": self.learning_rate,
                "hp/batch_size": self.batch_size,
            },
        )

    def prepare_data(self):
        # add check if data exists
        if (
            os.path.exists(os.path.join(self.data_dir, "Zebtrain.npy"))
            is False
        ):
            classification_dict = {str(n): n for n in range(23)}

            train_dir = os.path.join(self.data_dir, "train_files")
            test_dir = os.path.join(self.data_dir, "test_files")

            train_files = [
                os.path.join(train_dir, file)
                for file in os.listdir(train_dir)
                if ".h5" in file
            ]

            test_files = [
                os.path.join(test_dir, file)
                for file in os.listdir(test_dir)
                if ".h5" in file
            ]
            val_files = []

            preprocess = PreProcessing(
                train_files,
                test_files,
                val_files,
                classification_dict=classification_dict,
                n_nodes=19,
                center_node=13,
                animation=False,
                animal="Zeb",
            )
            preprocess.create_training_dataset(
                "train",
                center_all=True,
                align=True,
                rotate_jitter=False,
                ideal_sample_no=3000,
            )
            preprocess.create_training_dataset(
                "test",
                center_all=True,
                align=True,
                rotate_jitter=False,
                ideal_sample_no=3000,
            )  # print(np.unique(preprocess.labels, return_counts = True))
        else:
            print("data already exists")

    def setup(self, stage=None):
        target_transform = Lambda(
            lambda y: torch.zeros(
                self.num_classes, dtype=torch.float
            ).scatter_(0, torch.tensor(y), value=1)
        )
        self.train_data = ZebData(
            os.path.join(self.data_dir, "Zebtrain.npy"),
            os.path.join(self.data_dir, "Zebtrain_labels.npy"),
            target_transform=target_transform,
            augment=self.augment,
            ideal_sample_no=self.ideal_sample_no,
            shift=self.shift,
        )

        self.test_data = ZebData(
            os.path.join(self.data_dir, "Zebtest.npy"),
            os.path.join(self.data_dir, "Zebtest_labels.npy"),
            target_transform=target_transform,
        )

        if stage == "fit" or stage is None:
            train_length = int(
                len(self.train_data.labels) * 0.8
            )  # add augmentation step here
            val_length = (
                len(self.train_data.labels) - train_length
            )  # int(len(self.train_data.labels) * 0.2)

            # Split data into train and test
            self.pose_train, self.pose_val = random_split(
                self.train_data,
                [train_length, val_length],
                generator=torch.Generator().manual_seed(42),
            )

            if self.ideal_sample_no is not None:
                print("Dynamically Augmenting DataSet")
                # create two new blank ZebData
                self.pose_train_data = ZebData(
                    ideal_sample_no=self.ideal_sample_no
                )  # this will be augmented
                self.pose_val_data = ZebData()

                # assign data and labels to these new ones using train and val indices
                self.pose_train_data.data = self.pose_train.dataset.data[
                    self.pose_train.indices
                ]
                self.pose_train_data.labels = self.pose_train.dataset.labels[
                    self.pose_train.indices
                ]

                self.pose_val_data.data = self.pose_val.dataset.data[
                    self.pose_val.indices
                ]
                self.pose_val_data.labels = self.pose_val.dataset.labels[
                    self.pose_val.indices
                ]

                # dynamically augment
                self.pose_train_data.dynamic_augmentation()

                # reassign variables
                self.pose_train = self.pose_train_data
                self.pose_val = self.pose_val_data

        if stage == "test" or stage is None:
            self.pose_test = self.test_data

    def train_dataloader(self):
        return DataLoader(
            self.pose_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.pose_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.pose_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def extract_feature(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(
            in_channels, out_channels, kernel_size[1]
        )

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
