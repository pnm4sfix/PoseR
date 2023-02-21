import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional.classification.accuracy import accuracy


from .._loader import ZebData

# from preprocessing import PreProcessing


# Create CNN Model
class C3D(LightningModule):
    def __init__(
        self,
        num_class,
        num_channels,
        data_cfg,
        hparams,
        batch_size=0,
        num_workers=0,
        dropout=0,
    ):
        # super(C3D, self).__init__()
        super().__init__()
        self.num_workers = num_workers
        self.num_channels = num_channels

        self.data_dir = data_cfg["data_dir"]
        self.augment = data_cfg["augment"]
        self.ideal_sample_no = data_cfg["ideal_sample_no"]
        self.shift = data_cfg["shift"]

        try:
            self.transform = data_cfg["transform"]
        except:
            self.transform = None

        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size

        self.num_classes = num_class
        self.save_hyperparameters("hparams")

        ### Define architecture
        self.conv_config = dict(type="Conv3d")

        kernel_size = (3, 3, 3)
        padding = (1, 1, 1)
        # c3d_conv_param = dict(
        #    kernel_size=(3, 3, 3),
        #    padding=(1, 1, 1),
        #    conv_cfg=self.conv_cfg,
        #    norm_cfg=self.norm_cfg,
        #    act_cfg=self.act_cfg)

        # channel = nodes?

        self.conv1a = self._conv_layer_set(
            self.num_channels, 32, kernel_size, padding
        )  # ConvModule(3, 64, **c3d_conv_param)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = self._conv_layer_set(
            32, 64, kernel_size, padding
        )  # ConvModule(64, 128, **c3d_conv_param)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = self._conv_layer_set(
            64, 128, kernel_size, padding
        )  # ConvModule(128, 256, **c3d_conv_param)
        self.conv3b = self._conv_layer_set(
            128, 256, kernel_size, padding
        )  # ConvModule(256, 256, **c3d_conv_param)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = self._conv_layer_set(
            256, 256, kernel_size, padding
        )  # ConvModule(256, 512, **c3d_conv_param)
        # self.conv4b = self._conv_layer_set(512, 512, kernel_size, padding)#ConvModule(512, 512, **c3d_conv_param)
        self.pool4 = nn.MaxPool3d(
            kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)
        )

        # self.conv5a = self._conv_layer_set(512, 512, kernel_size, padding)#ConvModule(512, 512, **c3d_conv_param)
        # self.conv5b = self._conv_layer_set(512, 512, kernel_size, padding)#ConvModule(512, 512, **c3d_conv_param)
        # self.pool5 = nn.MaxPool3d(
        #    kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        ###

        # self.conv_layer1 = self._conv_layer_set(3, 32)

        # self.conv_layer2 = self._conv_layer_set(32, 64)

        self.fc1 = nn.Linear(12800, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, self.num_classes)
        self.relu = nn.ReLU()
        # self.batch=nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def _conv_layer_set(self, in_c, out_c, kernel_size, padding):
        # conv_layer = nn.Sequential(
        conv_layer = nn.Conv3d(
            in_c, out_c, kernel_size=kernel_size, padding=padding
        )
        # nn.LeakyReLU(),
        # nn.MaxPool3d((2, 2, 2)),
        # )
        return conv_layer

    def configure_optimizers(self):
        # Make sure to filter the parameters based on `requires_grad`

        # return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr = self.learning_rate)
        return torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
            momentum=0.9,
        )
        # optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        # return optimizer

    def training_step(self, batch, batch_idx):
        # Make sure dataloaders are on cuda
        x, y = batch
        output = self(x)
        print(output)
        loss = F.cross_entropy(output, y)
        preds = torch.argmax(output, dim=1)
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.num_classes
        )
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
        acc = accuracy(
            preds, y, task="multiclass", num_classes=self.num_classes
        )

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

    def setup(self, stage=None):
        print(f"STAGE IS {stage}")
        if stage == "predict":
            pass
        else:
            target_transform = None
            # target_transform = Lambda(
            #    lambda y: torch.zeros(
            #        self.num_classes, dtype=torch.float
            #    ).scatter_(0, torch.tensor(y), value=1)
            # )
            self.train_data = ZebData(
                os.path.join(self.data_dir, "Zebtrain.npy"),
                os.path.join(self.data_dir, "Zebtrain_labels.npy"),
                target_transform=target_transform,
                augment=self.augment,
                ideal_sample_no=self.ideal_sample_no,
                shift=self.shift,
                transform=self.transform,
            )

            self.test_data = ZebData(
                os.path.join(self.data_dir, "Zebtest.npy"),
                os.path.join(self.data_dir, "Zebtest_labels.npy"),
                target_transform=target_transform,
                transform=self.transform,
            )

            if stage == "fit" or stage is None:
                targets = self.train_data.labels
                train_idx, val_idx = train_test_split(
                    np.arange(len(targets)),
                    test_size=0.1765,
                    shuffle=True,
                    stratify=targets,
                    random_state=42,
                )

                # train_length = int(len(self.train_data.labels) *0.8) # add augmentation step here
                # val_length = len(self.train_data.labels)-train_length #int(len(self.train_data.labels) * 0.2)

                # Split data into train and test
                self.pose_train, self.pose_val = Subset(
                    self.train_data, train_idx
                ), Subset(self.train_data, val_idx)

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
                    self.pose_train_data.labels = (
                        self.pose_train.dataset.labels[self.pose_train.indices]
                    )

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        print(len(batch))
        print(batch[0].shape)
        return self(batch[0])

    def forward(self, x):
        #
        """
        Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
                the size of x is (num_batches, 3, 16, 112, 112).
        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.relu(self.conv1a(x))
        x = self.pool1(x)

        x = self.relu(self.conv2a(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        # x = self.conv4b(x)
        x = self.pool4(x)

        # x = self.conv5a(x)
        # x = self.conv5b(x)
        # x = self.pool5(x)

        x = x.flatten(start_dim=1)
        # x = x.view(x.size(0) , -1)
        # x = x.view(-1, 12800)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        logits = self.fc3(x)

        probs = logits  # self.softmax(logits)

        return probs

        # Set 1
        # out = self.conv_layer1(x)
        # out = self.conv_layer2(out)
        # out = out.view(out.size(0), -1)
        # out = self.fc1(out)
        # out = self.relu(out)
        # out = self.batch(out)
        # out = self.drop(out)
        # out = self.fc2(out)

        # return out
