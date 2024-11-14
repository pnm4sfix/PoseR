r"""Copyright [@misc{mmskeleton2019,
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
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchmetrics.functional.classification.accuracy import accuracy
from torcheval.metrics.functional import multiclass_auprc
from torcheval.metrics import MulticlassAUPRC   
from torcheval.metrics.functional import binary_accuracy
from torcheval.metrics import BinaryAUPRC
from torcheval.metrics import MulticlassAccuracy, BinaryAccuracy
import time

# from torchmetrics.regression import R2Score
from torcheval.metrics.functional import r2_score

# import sys
# sys.path.insert(1, "../")
from .._loader import ZebData
from . import ConvTemporalGraphical, Graph

# from preprocessing import PreProcessing


def zero(x):
    return 0


def iden(x):
    return x

# New code from Pierce Mullen
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
        **kwargs,
    ):
        super().__init__()
        # self.hparams.update(hparams)
        try:
            self.save_hyperparameters()
        except:
            pass
        self.num_workers = num_workers
        self.data_dir = data_cfg["data_dir"]
        self.augment = data_cfg["augment"]
        self.ideal_sample_no = data_cfg["ideal_sample_no"]
        self.shift = data_cfg["shift"]

        try:
            self.regress = data_cfg["regress"]
        except:
            self.regress = False

        try:
            self.soft = data_cfg["softmax"]
        except:
            self.soft = False

        try:
            self.transform = data_cfg["transform"]
        except:
            self.transform = None

        try:
            self.labels_to_ignore = data_cfg["labels_to_ignore"]
        except:
            self.labels_to_ignore = None

        try:
            self.label_dict = data_cfg["label_dict"]
        except:
            self.label_dict = None

        try:
            self.calc_class_weights = data_cfg["calc_class_weights"]
        except:
            self.calc_class_weights = False

        try:
            self.center_node = graph_cfg["center_node"]
        except:
            self.center_node = 0

        try:
            self.T = data_cfg["T2"]
        except:
            self.T = None

        try:
            self.head_node = data_cfg["head"]
        except:
            self.head_node = 0

        try:
            self.weighted_random_sampler = data_cfg["weighted_random_sampler"]
        except:
            self.weighted_random_sampler = None

        try:
            self.binary = data_cfg["binary"]
            self.binary_class = data_cfg["binary_class"]
            print(f"Binary is {self.binary}")
        except:
            self.binary = False
            self.binary_class = None
        try:
           self.preprocess_frame = data_cfg["preprocess_frame"]
           self.window_size = data_cfg["window_size"]
        except:
            self.preprocess_frame = False
            self.window_size = None


        self.learning_rate = hparams.learning_rate
        self.batch_size = hparams.batch_size
        self.dropout = hparams.dropout
        
        if self.binary:
            self.num_classes = 1
        else:
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
                    in_channels, 128, kernel_size, 1, residual=False, **kwargs0
                ),
                st_gcn_block(
                    128, 128, kernel_size, 1, dropout=self.dropout, **kwargs
                ),  # remove to trim
                st_gcn_block(128, 256, kernel_size, 2, **kwargs),
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
        self.fcn = nn.Conv2d(256, self.num_classes, kernel_size=1)
        if self.soft:
            self.softmax = nn.Softmax(dim=1)

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
        #if self.soft:
        #    x = self.softmax(x)
        # x = self.softmax(x) # don't need as we use cross entropy loss
        return x

    def configure_optimizers(self):
        # Make sure to filter the parameters based on `requires_grad`

        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )
        # optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        # return optimizer

    def training_step(self, batch, batch_idx):
        # Make sure dataloaders are on cuda
        x, y = batch
        output = self(x)
        if self.regress:
            loss = F.l1_loss(
                output, y
            )  # mse loss is double where it expects a float
            preds = output
            # r2score = R2Score() #R2Score(num_outputs=self.num_classes, multioutput='raw_values').to(self.device)
            acc = r2_score(preds, y)

        elif self.binary:
            if self.calc_class_weights:
                no_label_weight = self.class_weights[0]
                label_weight = self.class_weights[1]
                weights = torch.tensor([no_label_weight if t == 0 else label_weight for t in y], dtype=torch.float32).to(self.device)
            else:
                weights = None
            loss = F.binary_cross_entropy_with_logits(output.flatten(), y, weight = weights)
            preds = torch.sigmoid(output.flatten())
            acc = binary_accuracy(preds, y)
        else:
            loss = F.cross_entropy(output, y, weight=self.class_weights)
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
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        self.log("elapsed_time", elapsed_time, prog_bar=True, logger=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        if self.regress:
            loss = F.l1_loss(output, y)  # l1
            preds = output
            # r2score = R2Score().to(self.device)
            acc = r2_score(preds, y)

        elif self.binary:
            loss = F.binary_cross_entropy_with_logits(output.flatten(), y)
            preds = torch.sigmoid(output.flatten())
            #acc = binary_accuracy(preds, y)
            self.auprc.update(output.flatten(), y)
            self.acc.update(preds, y)
        else:
            loss = F.cross_entropy(output, y)
            preds = torch.argmax(output, dim=1)
            #acc = accuracy(
            #    preds, y, task="multiclass", num_classes=self.num_classes
            #)
            self.auprc.update(output, y)
            self.acc.update(preds, y)
            #auprc = multiclass_auprc(output, y, num_classes=self.num_classes)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            logger=True,
            on_epoch=True,
        )

        #self.log(
        #    "auprc",
        #    auprc,
        #    prog_bar=True,
        #    sync_dist=True,
        #    logger=True,
        #    on_epoch=True,
        #)

        #self.log("val_acc", acc, prog_bar=True, logger=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        if self.regress:
            loss = F.l1_loss(output, y)
            preds = output
            # r2score = R2Score().to(self.device)
            acc = r2_score(preds, y)

        elif self.binary:
            
            loss = F.binary_cross_entropy_with_logits(output.flatten(), y)
            preds = torch.sigmoid(output.flatten())
            #acc = binary_accuracy(preds, y)
            self.auprc.update(output.flatten(), y)
            self.acc.update(preds, y)
        else:
            loss = F.cross_entropy(output, y)
            preds = torch.argmax(output, dim=1)
            self.auprc.update(output, y)
            self.acc.update(preds, y)
            #acc = accuracy(
            #    preds, y, task="multiclass", num_classes=self.num_classes
            #)
            # acc3 = accuracy(preds, y, task="multiclass", num_classes=self.num_classes, top_k = 3)

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            logger=True,
            on_epoch=True,
        )
        #self.log("val_acc", acc, prog_bar=True, logger=True, on_epoch=True)
        # self.log("val_acc_top3", acc3, prog_bar = True)

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
        self.start_time = time.time()

    def on_train_end(self):
        self.end_time = time.time()
        self.logger.log_metrics(
            {"training_time": self.end_time - self.start_time})
        print(f"Total training time: {self.end_time - self.start_time:.4f} seconds")
        #self.log("time_elapsed", self.end_time - self.start_time, logger = True, on_epoch = False, sync_dist = True)

    def on_validation_start(self):
        self.val_start_time = time.time()

    def on_validation_end(self):
        self.val_end_time = time.time()
        self.logger.log_metrics(
            {"validation_time": self.val_end_time - self.val_start_time})
        #self.log("time_elapsed", self.end_time - self.start_time, logger = True, on_epoch = False, sync_dist = True)

    def on_test_start(self):
        self.start_time = time.time()

    def on_test_end(self):
        self.end_time = time.time()
        self.logger.log_metrics(
            {"test_time": self.end_time - self.start_time})
        #self.log("time_elapsed", self.end_time - self.start_time, logger = True, on_epoch = False, sync_dist = True)

    def on_predict_start(self):
        self.start_time = time.time()

    def on_predict_end(self):
        self.end_time = time.time()
        self.logger.log_metrics(
            {"prediction_time": self.end_time - self.start_time})
        print(f"Total prediction time: {self.end_time - self.start_time:.4f} seconds")
        #self.log("time_elapsed", self.end_time - self.start_time, logger = True, on_epoch = False, sync_dist = True)
        

    def on_validation_epoch_start(self):
        # Create the metric at the start of each validation epoch
        if self.binary:
            self.auprc = BinaryAUPRC()  # For binary
            self.acc = BinaryAccuracy()
        else:
            self.auprc = MulticlassAUPRC(num_classes = self.num_classes)  # For multiclass
            self.acc = MulticlassAccuracy(num_classes = self.num_classes)

    def on_validation_epoch_end(self):
        # Log the metric value at the end of the validation epoch
        self.log("auprc", self.auprc.compute(), prog_bar=True, logger=True, on_epoch=True, sync_dist = True)
        self.log("val_acc", self.acc.compute(), prog_bar=True, logger=True, on_epoch=True, sync_dist = True)

    def on_test_epoch_start(self):
        # Create the metric at the start of each test epoch
        if self.binary:
            self.auprc = BinaryAUPRC()
            self.acc = BinaryAccuracy()
        else:
            self.auprc = MulticlassAUPRC(num_classes = self.num_classes)
            self.acc = MulticlassAccuracy(num_classes = self.num_classes)

    def on_test_epoch_end(self):
        # Log the metric value at the end of the test epoch
        self.log("auprc", self.auprc.compute(), prog_bar=True, logger=True, on_epoch=True, sync_dist = True)
        self.log("test_acc", self.acc.compute(), prog_bar=True, logger=True, on_epoch=True, sync_dist = True)

    




    # def prepare_data(self):
    #    # add check if data exists
    #    if os.path.exists(os.path.join(self.data_dir, "Zebtrain.npy")) is False:
    #        classification_dict = {str(n):n for n in range (23)}
    #
    #            train_dir = os.path.join(self.data_dir, 'train_files')
    #            test_dir = os.path.join(self.data_dir, 'test_files')
    #
    #            train_files = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if '.h5' in file]
    #
    #            test_files = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if '.h5' in file]
    #            val_files = []
    #
    #
    #           #preprocess = PreProcessing(train_files, test_files, val_files, classification_dict = classification_dict, n_nodes=19, center_node = 13, animation = False,  animal = "Zeb")
    #          #preprocess.create_training_dataset("train", center_all = True, align = True, rotate_jitter = False, ideal_sample_no = 3000)
    #         #preprocess.create_training_dataset("test", center_all = True, align = True, rotate_jitter = False, ideal_sample_no = 3000)#print(np.unique(preprocess.labels, return_counts = True))
    #        else:
    #            print("data already exists")

    def setup(self, stage=None):
        print(f"STAGE IS {stage}")
        if stage == "predict":
            pass
        elif stage == "validate":
            try: 
                self.val_data = ZebData(
                os.path.join(self.data_dir, "Zebval.npy"),
                os.path.join(self.data_dir, "Zebval_labels.npy"),
                target_transform=target_transform,
                #augment=self.augment,
                ideal_sample_no=self.ideal_sample_no,
                shift=self.shift,
                transform=self.transform,
                labels_to_ignore=self.labels_to_ignore,
                label_dict=self.label_dict,
                regress=self.regress,
                T = self.T,
                center_node=self.center_node,
                head_node = self.head_node,
                binary = self.binary,
                binary_class = self.binary_class,
                preprocess_frame=self.preprocess_frame,
                window_size=self.window_size
                )
                self.pose_val = self.val_data
                print("validating")
            except:
                self.val_data = None
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
                labels_to_ignore=self.labels_to_ignore,
                label_dict=self.label_dict,
                regress=self.regress,
                T = self.T,
                center_node=self.center_node,
                head_node=self.head_node,
                binary = self.binary,
                binary_class = self.binary_class,
                preprocess_frame=self.preprocess_frame,
                window_size=self.window_size
            )

            try: 
                self.val_data = ZebData(
                os.path.join(self.data_dir, "Zebval.npy"),
                os.path.join(self.data_dir, "Zebval_labels.npy"),
                target_transform=target_transform,
                #augment=self.augment,
                ideal_sample_no=self.ideal_sample_no,
                shift=self.shift,
                transform=self.transform,
                labels_to_ignore=self.labels_to_ignore,
                label_dict=self.label_dict,
                regress=self.regress,
                T = self.T,
                center_node=self.center_node,
                head_node = self.head_node,
                binary = self.binary,
                binary_class = self.binary_class,
                preprocess_frame=self.preprocess_frame,
                window_size=self.window_size
                )
            except:
                self.val_data = None

            self.test_data = ZebData(
                os.path.join(self.data_dir, "Zebtest.npy"),
                os.path.join(self.data_dir, "Zebtest_labels.npy"),
                target_transform=target_transform,
                transform=self.transform,
                labels_to_ignore=self.labels_to_ignore,
                label_dict=self.label_dict,
                regress=self.regress,
                T = self.T,
                center_node=self.center_node,
                head_node=self.head_node,
                binary = self.binary,
                binary_class = self.binary_class,
                preprocess_frame=self.preprocess_frame,
                window_size=self.window_size
            )



            if stage == "fit" or stage is None:
                targets = self.train_data.labels

                if self.val_data is None:
                    if self.regress:
                        train_idx, val_idx = train_test_split(
                            np.arange(len(targets)),
                            test_size=0.1765,
                            shuffle=True,
                            random_state=42,
                        )
                    else:
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

                    """if self.ideal_sample_no is not None:
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
                        self.pose_val = self.pose_val_data"""

                    if self.calc_class_weights:
                        self.class_weights = self.train_data.get_class_weights()
                        print(f"Class weights are {self.class_weights}")

                        # put self.class_weights on cuda
                        self.class_weights = self.class_weights.cuda()

                    else:
                        self.class_weights = None
                else:
                    """if self.ideal_sample_no is not None:
                        self.pose_train_data = ZebData(
                            ideal_sample_no=self.ideal_sample_no
                        ) 

                        # assign data and labels to these new ones using train and val indices
                        self.pose_train_data.data = self.train_data.data
                        self.pose_train_data.labels = self.train_data.labels

                        # dynamically augment
                        self.pose_train_data.dynamic_augmentation()

                        # reassign variables
                        self.pose_train = self.pose_train_data"""


                    if self.calc_class_weights:
                        self.class_weights = self.train_data.get_class_weights()
                        print(f"Class weights are {self.class_weights}")

                        # put self.class_weights on cuda
                        self.class_weights = self.class_weights.cuda()

                    else:
                        self.class_weights = None

                    # limit class sizes
                    """if self.ideal_sample_no is not None:
                        # get label indexes for each label and subset those indices to get ideal sample no
                        # get unique labels
                        if self.ideal_sample_no == -1:
                            ideal_sample_no = np.int(np.median(
                                np.unique(self.train_data.labels, return_counts=True)[1]
                            ))
                        print("Limiting train class size to {}".format(ideal_sample_no))
                        unique_labels = np.unique(self.train_data.labels)
                        # size limited indexes
                        label_idxs = []
                        # loop through labels
                        for v in unique_labels:
                            label_idx = np.where(self.train_data.labels == v)[0]
                            label_count = label_idx.shape[0]
                            if label_count > ideal_sample_no:
                                # subset to ideal sample no
                                subset = np.random.choice(
                                    label_idx, ideal_sample_no, replace=False
                                )
                                label_idxs.append(subset)
                            else:
                                label_idxs.append(label_idx)

                        label_idxs = np.concatenate(label_idxs)
                
                        self.train_data = Subset(self.train_data, label_idxs)

                        if self.ideal_sample_no == -1:
                            ideal_sample_no = np.int(np.median(
                                np.unique(self.val_data.labels, return_counts=True)[1]
                            ))
                        print("Limiting val class size to {}".format(ideal_sample_no))
                        unique_labels = np.unique(self.val_data.labels)
                        # size limited indexes
                        label_idxs = []
                        # loop through labels
                        for v in unique_labels:
                            label_idx = np.where(self.val_data.labels == v)[0]
                            label_count = label_idx.shape[0]
                            if label_count > ideal_sample_no:
                                # subset to ideal sample no
                                subset = np.random.choice(
                                    label_idx, ideal_sample_no, replace=False
                                )
                                label_idxs.append(subset)
                            else:
                                label_idxs.append(label_idx)

                        label_idxs = np.concatenate(label_idxs)
                
                        self.val_data = Subset(self.val_data, label_idxs)"""




                    self.pose_train = self.train_data
                    self.pose_val = self.val_data



                    

            if stage == "test" or stage is None:
                self.pose_test = self.test_data

    def train_dataloader(self):
        if self.weighted_random_sampler:
            if isinstance(self.pose_train, Subset):
                class_weights = self.pose_train.dataset.get_class_weights()
                class_weight_map = {k:v for k,v in enumerate(class_weights)}
                sample_weights = [class_weights[self.pose_train.dataset.labels[idx].item()] for idx in self.pose_train.indices]

            else:
                sample_weights = self.pose_train.get_sample_weights()
            if self.ideal_sample_no is not None:
                num_samples = self.ideal_sample_no
            else:
                num_samples = len(self.pose_train)
            print("Weighted Random Sampler of size {}".format(num_samples))
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=num_samples,
                replacement=True,
            )
            print("Using Weighted Random Sampler")
            shuffle = False
            
        else:
            sampler = None
            shuffle = True
            print("Not using Weighted Random Sampler")

        return DataLoader(
            self.pose_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            sampler=sampler)
    

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # print(len(batch))
        # print(batch[0].shape)
        return self(batch[0])

### Original code from Sijie Yan, Yuanjun Xiong, Jingbo Wang, Dahua Lin (2019)
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
