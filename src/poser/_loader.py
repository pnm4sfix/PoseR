from cProfile import label
import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight
import psutil
import os

try:
    import cupy as cp
except:
    print("Couldnt import cupy")


# Load and evalulate MMS model
# Model class must be defined somewhere
# model = torch.load("./model/st_gcn.kinetics-6fa43f73.pth")
# model.eval()

# Replace last layer with own classifier

# Create dataset

# Create dateset loader

# Create optimiser

# optimizer = optim.ADAM(model.parameters(), lr = 0.001, momentum =0.9)

# Split data into

# Finetuning
# Freeze all the parameters in the network
# for param in model.parameters():
#    param.requires_grad = False
# replace last layer (this example is resnet)
# model.fc = nn.Linear(512, 10)
# Optimize only the classifier
# optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)


class ZebData(torch.utils.data.Dataset):

    """Class for test data - because data isnt large just load data at init"""

    def __init__(
        self,
        data_file=None,
        label_file=None,
        transform=None,
        target_transform=None,
        ideal_sample_no=None,
        augment=False,
        shift=False,
        labels_to_ignore=None,
        label_dict=None,
        regress=False,
        center_node=0,
        head_node = 0, 
        T = None,
        lazy = True,
        binary = False,
        binary_class = None,
        preprocess_frame = False,
        window_size = None
    ):
        self.ideal_sample_no = ideal_sample_no
   
        self.transform = transform
        self.target_transform = target_transform
        self.center_node = center_node
        self.head_node = head_node
        self.T= T # final padded T
        self.augment = augment
        self.preprocess_frame = preprocess_frame
        self.window_size = window_size
        print(f"Augment is {self.augment}")

        if data_file is not None:
            # check size of file
            filesize = os.path.getsize(data_file)/1e9
            # Get virtual memory details
            memory_info = psutil.virtual_memory()

            # Available memory in bytes
            available_memory = memory_info.available / 1e9

            if filesize > available_memory:
                print("File too large for memory, using mmap mode")
                self.data = np.load(data_file, mmap_mode = "r")
            else:
                self.data = np.load(data_file)

            # catch incorrectly loaded shape
            if self.data.shape[0] == 1:
                self.data = self.data.reshape(*self.data.shape[1:])

            # if data is 4D array C, T, V, M
            if self.data.ndim == 4:
                self.C, self.T0, self.V, self.M = self.data.shape
                # print statement about number of channels, frames , nodes and individuals
                print(f"Data shape is {self.data.shape} C is {self.C}, T is {self.T0}, V is {self.V}, M is {self.M}")

                
            # elif data is 5D array N, C, T, V, M
            elif self.data.ndim == 5:
                self.N, self.C, self.T0, self.V, self.M = self.data.shape
                print(f"Data shape is {self.data.shape} N is {self.N} C is {self.C}, T is {self.T0}, V is {self.V}, M is {self.M}")



            if regress:
                self.labels = np.load(label_file).astype("float64")

            elif binary:
                self.labels = np.load(label_file).astype("float64")
                if self.labels.shape[0] == 1:
                    self.labels = self.labels.reshape(*self.labels.shape[1:])

                if binary_class is not None:
                    label_filter = np.isin(self.labels, binary_class)
                    self.labels[~label_filter] = 0
                    self.labels[label_filter] = 1
                    print(f"Binary class is {binary_class}")
                    print(f"Binary labels are {np.unique(self.labels)}")

            else:
                # catch incorrectly loaded shape
                self.labels = np.load(label_file).astype("int64")
                if self.labels.shape[0] == 1:
                    self.labels = self.labels.reshape(*self.labels.shape[1:])

                # drop junk 0 cluster
                # self.data = self.data[self.labels>0]
                # self.labels = self.labels[self.labels>0]
                print(
                    f"Dataset breakdown is {pd.Series(self.labels).value_counts()}"
                )

                if labels_to_ignore is not None:
                    label_filter = np.isin(self.labels, labels_to_ignore)
                    self.labels = self.labels[~label_filter]
                    self.data = self.data[~label_filter]
                    print(f"Ignoring Labels {labels_to_ignore}")
                    print(
                        f"Filtered data contains labels {np.unique(self.labels)}"
                    )

                if label_dict is None:
                    print("No label dict")
                    mapping = {
                        k: v for v, k in enumerate(np.unique(self.labels))
                    }
                    for k, v in mapping.items():
                        self.labels[self.labels == k] = v
                
                elif label_dict is not None:
                    new_labels = np.zeros_like(self.labels)
                    mapping = label_dict
                    for k, v in mapping.items():
                        print("original label is {} and new label is {}".format(k, v))
                        new_labels[self.labels == k] = v
                    self.labels = new_labels
                    print("Labels already mapped during saving")
                    # mapping = label_dict
                    # semantic: value

                print(f"label mapping is {mapping}")
                print(f"Unique labels are {np.unique(self.labels)}")
            
            
                        
            # if augment:

            #    self.dynamic_augmentation()

            if shift:
                # move into positive space

                self.data = self.data + np.array([50, 50])

        elif data_file is None:
            # data can be added manually later
            self.data = None
            self.labels = None

        if self.transform == "heatmap":
            x_min = -3  # self.data[:,0].min()
            x_max = 3  # self.data[:,0].max()

            y_min = -3  # self.data[:, 1].min()
            y_max = 3  # self.data[:, 1].max()

            self.H = self.W = 64
            x = cp.linspace(x_min, x_max, num=self.W, dtype="float32")
            y = cp.linspace(y_min, y_max, num=self.H, dtype="float32")
            self.xv, self.yv = cp.meshgrid(x, y, indexing="xy")
            self.xv_rs = self.xv.reshape((*self.xv.shape, 1))
            self.yv_rs = self.yv.reshape((*self.yv.shape, 1))
            print(
                f"x min is {x_min}, x max is {x_max}, y min is {y_min} y max is {y_max}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # if data is from raw CTVM it needs to be windowed appropriately first
        if self.preprocess_frame:
            behaviour = self.preprocess_frames(idx, self.window_size) # assume data is CTVM shape
            label = self.labels[idx]
        else:
            behaviour = self.data[idx].copy()
            label = self.labels[idx]
        

        if self.transform is not None:
            if "flipy" in self.transform:
                behaviour = self.flipy(behaviour)

            if "center" in self.transform:
                behaviour = self.center_all(behaviour, self.center_node)

            if (self.transform == "align") | ("align" in self.transform):
                behaviour = self.align(behaviour)

            if "pad" in self.transform:
                behaviour = self.pad(behaviour, self.T)

            if "pad_flip" in self.transform:
                behaviour = self.pad_flip(behaviour, self.T)

            if self.augment == True:
                behaviour = self.random_augmentation(behaviour)
                

            if self.transform == "heatmap":
                # print("transforming")
                # behaviour = self.convert_bout_to_heatmap(behaviour, self.W, self.H, self.xv, self.yv)
                # behaviour = torch.from_numpy(behaviour).to(torch.float32)
                behaviour = torch.as_tensor(
                    self.cp_bout_to_heatmap(
                        cp.asarray(behaviour[:, ::4]),
                        self.W,
                        self.H,  # get every 3rd timepoint to reduce size
                        self.xv_rs,
                        self.yv_rs,
                    ),
                    device="cpu",
                    dtype=torch.float32,
                )

        if self.transform is None:
            behaviour = torch.from_numpy(behaviour).to(torch.float32)
        if type(behaviour) != torch.Tensor:
            behaviour = torch.from_numpy(behaviour).to(torch.float32)
        # if self.target_transform is not None:
        # label = self.target_transform(label)
        if label.dtype != "float64":
            label = torch.tensor(label).long()
        else:
            label = torch.tensor(label)
        return behaviour, label

    

    def preprocess_frames(self, frame, window_size):
        # assume data is CTVM if 4D
        window_half_width = int(window_size // 2)
        
        t0 = frame-window_half_width
        t1 = frame+window_half_width

        # define boundary conditions
        if t0 < 0:
            left_pad = np.zeros((self.C, np.abs(t0), self.V, self.M))
            t0 = 0
        else:
            left_pad = None
        if t1 > self.T0:
            right_pad = np.zeros((self.C, t1-self.T0, self.V, self.M))
            
            t1 = self.T0
        else:
            right_pad = None

        bhv = self.data[:, t0:t1].copy() # C, T, V, M
           

        if left_pad is not None:
            bhv = np.concatenate([left_pad, bhv], axis=1)
                
        if right_pad is not None:
            bhv = np.concatenate([bhv, right_pad], axis=1)

        return bhv
                

    def flipy(self, bhv):
        # Negate only the y-coordinates (assuming y is at index 1)
        bhv[1, :, :, :] *= -1
        return bhv


    def align(self, bhv_rs):
        # assumes bout already centered - add check if nose node is <
        # first node coords is vector from center
        nose_node_vector = bhv_rs[0:2, 0, self.head_node]
        nose_node_mag = np.linalg.norm(nose_node_vector)
        center_vector = np.array([0, 1]).reshape(
            -1, 1
        )  # changed from nose_node_mag
        center_vector_mag = np.linalg.norm(center_vector)
        # cosine rule to find angle between nose node vector and center vector
        # theta  = (np.arccos(np.dot(nose_node_vector.flatten(), center_vector.flatten())/(nose_node_mag*center_vector_mag)))
        # if nose_node_vector[0] < 0:
        # theta = -theta
        # if nose_node_vector[0] < 0:
        # theta = np.arctan2(-nose_node_vector[0], nose_node_vector[1])[0]

        # else:
        theta = np.arctan2(nose_node_vector[0], nose_node_vector[1])[
            0
        ]  # using arctan2 this way gives angle from (0, 1)

        # counterclockwise rotation
        if theta < 0:
            theta = theta + (
                2 * np.pi
            )  # important to make sure counter rotates properly when given signed angle

        c, s = np.cos(theta), np.sin(theta)
        # rotation matrix to use to transform coordinate space

        R = np.array(((c, -s), (s, c)))
        # transform with rotation matrix
        bhv_rs[0:2, :] = np.dot(R, bhv_rs[0:2, :].reshape((2, -1))).reshape(
            bhv_rs[0:2, :].shape
        )
        return bhv_rs

    def center_all(self, bout, center_node):
        center_node_xy = bout[0:2, :, center_node]

        center_node_xy = center_node_xy.reshape(
            center_node_xy.shape[0],
            center_node_xy.shape[1],
            -1,
            center_node_xy.shape[2],
        )  # reshaping to match shape of main array for subtraction

        centered_bout = bout.copy()
        centered_bout[0:2] = centered_bout[0:2] - center_node_xy
        return centered_bout

    def center_first(self, bout, center_node):
        center_node_xy = bout[0:2, 0, center_node]
        center_node_xy = center_node_xy.reshape(2, -1, 1)

        center_node_xy = center_node_xy.reshape(
            center_node_xy.shape[0],
            center_node_xy.shape[1],
            -1,
            center_node_xy.shape[2],
        )  # reshaping to match shape of main array for subtraction

        centered_bout = bout.copy()
        centered_bout[0:2] = centered_bout[0:2] - center_node_xy
        return centered_bout

    def pad(self, bout, new_T):
        pose = bout
        padding = (
            new_T - pose.shape[1]
        )  # difference between standard T =50 and the length of actual sequence
        ratio = padding / pose.shape[1]
        if ratio > 1:
            bhv_pad = np.concatenate((pose, pose), axis=1)

            for r in range(int(ratio) - 1):
                bhv_pad = np.concatenate((bhv_pad, pose), axis=1)

            diff = new_T - bhv_pad.shape[1]
            bhv_pad = np.concatenate((bhv_pad, pose[:, :diff]), axis=1)

        elif (ratio <= 1) & (ratio > 0):
            diff = new_T - pose.shape[1]
            bhv_pad = np.concatenate((pose, pose[:, :diff]), axis=1)
        elif ratio <= 0:
            bhv_pad = pose[:, :new_T]

        return bhv_pad

    def pad_flip(self, bout, new_T):
        pose = bout
        padding = (
            new_T - pose.shape[1]
        )  # difference between standard T =50 and the length of actual sequence
        ratio = padding / pose.shape[1]
        if ratio > 1:
            bhv_pad = np.concatenate((pose, np.flip(pose, axis=1)), axis=1) # reverse on time dimension

            for r in range(int(ratio) - 1):
                # if even no flip, if not reverse - creates a sort of back and forth motion that will be periodic
                if r % 2 == 0:
                    bhv_pad = np.concatenate((bhv_pad, pose), axis=1)
                else:
                    bhv_pad = np.concatenate((bhv_pad, np.flip(pose, axis=1)), axis=1)

            diff = new_T - bhv_pad.shape[1]
            bhv_pad = np.concatenate((bhv_pad, pose[:, :diff]), axis=1)

        elif (ratio <= 1) & (ratio > 0):
            diff = new_T - pose.shape[1]
            bhv_pad = np.concatenate((pose, pose[:, :diff]), axis=1)
        elif ratio <= 0:
            bhv_pad = pose[:, :new_T]

        return bhv_pad


    # find angle from [0, 1]
    def angle_from_norm(self, coord):
        theta = np.degrees(
            np.arctan2(coord[0], coord[1])
        )  # arctan must be y then x
        return theta

    def get_heading_change(self, bout):
        nose = bout[:2, :, 0]
        last_nose = nose[:, -1]
        angle = self.angle_from_norm(last_nose)
        return angle

    def random_augmentation(self, bhv, numAug = 1):
        rotated = self.rotate_transform(bhv, numAug)[0]
        jittered = self.jitter_transform(rotated, numAug)[0]
        scaled = self.scale_transform(jittered, numAug)[0]
        sheared = self.shear_transform(scaled, numAug)[0]
        #rolled = self.roll_transform(sheared, numAug)[0]

        return sheared

    def dynamic_augmentation(self):
        drop_labels = []

        augmented_data = []
        augmented_labels = []
        bhv_idx = []
        for label in np.unique(self.labels):
            if label >= 0:
                filt = self.labels == label
                label_count = self.labels[filt].shape[0]
                label_subset = self.data[filt]

                augmented = np.zeros(
                    (
                        self.ideal_sample_no + label_count,
                        *label_subset.shape[1:],
                    )
                )

                if label_count < 1:  # this drops rare behaviours
                    # drop_labels.append(v)
                    pass

                elif label_count > self.ideal_sample_no:
                    augmented = label_subset[: self.ideal_sample_no]
                    augmented_data.append(augmented)
                    augmented_labels.append(
                        np.array([label] * augmented.shape[0]).flatten()
                    )
                    bhv_idx.append(np.arange(augmented.shape[0]))
                    print("sample greater than ideal sample no")
                    print(augmented.shape)
                    # break

                else:
                    ratio = self.ideal_sample_no / label_subset.shape[0]

                    augmentation_types = 5  # 6 if fragment working
                    remainder = int(ratio % augmentation_types)
                    numAug = int(ratio / augmentation_types)

                    # loop through behaviours in subset
                    for b in range(label_subset.shape[0]):
                        bhv = label_subset[b].copy()
                        rotated = self.rotate_transform(
                            bhv, numAug + remainder
                        )
                        jittered = self.jitter_transform(bhv, numAug)
                        scaled = self.scale_transform(bhv, numAug)
                        sheared = self.shear_transform(bhv, numAug)
                        rolled = self.roll_transform(bhv, numAug)
                        # fragment = self.fragment_transform(bhv, numAug)

                        # concatenate 5 augmentations and original
                        augmented = np.concatenate(
                            [
                                bhv.reshape(-1, *bhv.shape),
                                rotated,
                                jittered,
                                scaled,
                                sheared,
                                rolled,
                                # fragment,
                            ]
                        )
                        augmented_data.append(augmented)
                        augmented_labels.append(
                            np.array([label] * augmented.shape[0]).flatten()
                        )
                        bhv_idx.append(
                            np.array([b] * augmented.shape[0]).flatten()
                        )

        self.augmented_data = np.array(augmented_data)
        # self.data = self.augmented_data.reshape((-1, *augmented_data[0].shape[1:]))
        self.data = np.concatenate(self.augmented_data)
        self.augmented_labels = np.array(augmented_labels)
        self.labels = np.concatenate(self.augmented_labels)
        self.bhv_idx = np.concatenate(np.array(bhv_idx))

        print(
            f"New dataset characteristics {pd.Series(self.labels).value_counts()}"
        )

    def rotate_transform(self, behaviour, numAngles):
        """Rotates poses returning a set number of rotated poses.

        # N, C, T, V, M"""

        rotated = np.zeros((numAngles, *behaviour.shape))

        for angle_no in range(numAngles):
            # random angle between -50 and + 50
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

    def jitter_transform(self, behaviour, numJitter):
        """Adds noise to poses returning a set number of rotated poses.

        # N, C, T, V, M"""

        jittered = np.zeros((numJitter, *behaviour.shape))

        for jitter_no in range(numJitter):
            # random jitter between -5 and +5 pixels
            jitter = (np.random.random(behaviour[:2].shape) * 4) - 2

            jittered[jitter_no] = behaviour.copy()
            jittered[jitter_no, :2] = behaviour[:2] + jitter

        return jittered

    def scale_transform(self, behaviour, numScales):
        """Randomly scales poses"""

        scaled = np.zeros((numScales, *behaviour.shape))

        for scale_no in range(numScales):
            # create random scales between 0 and 3
            scale = np.random.random(1) * 3

            scaled[scale_no] = behaviour.copy()
            scaled[scale_no, :2] = behaviour[:2] * scale

        return scaled

    def shear_transform(self, behaviour, numShears):
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

    def roll_transform(self, behaviour, numRolls):
        rolled = np.zeros((numRolls, *behaviour.shape))

        for roll_no in range(numRolls):
            roll_x = np.random.randint(-20, 20, 1)[0]
            rolled[roll_no] = np.roll(behaviour, roll_x, axis=1)

        return rolled

    def fragment_transform(self, behaviour, numFragments):
        T = behaviour.shape[1]
        fragments = np.zeros((numFragments, *behaviour.shape))
        for fragment_no in range(numFragments):
            # create random scales between 0 and 3

            # define random start point around middle and length

            random_start = np.random.randint(0, T - 1, 1)[0]
            random_length = np.random.randint(10, 60, 1)[
                0
            ]  # random length between 10 and 40 frames

            fragment = behaviour[
                :, random_start : random_start + random_length, :, :
            ]

            # use pad function in this class to pad fragment bout to new T of self.T
            fragment = self.pad(fragment, T)

            fragments[fragment_no] = fragment
        return fragments

    # @staticmethod
    # @jit
    def convert_bout_to_heatmap(self, bout, W, H, xv, yv):
        C, T, V, M = np.shape(bout)  # .shape
        stack = np.empty((V, int(T / 2), W, H))

        for t_idx, t in enumerate(np.arange(0, int(T / 2) * 2, 2)):
            for k in np.arange(V):
                xk = bout[0, t, k, 0]  # N,C,T,V
                yk = bout[1, t, k, 0]
                ck = bout[2, t, k, 0]

                zz = (
                    np.exp(
                        -(((xv - xk) ** 2) + ((yv - yk) ** 2))
                        / (2 * (0.1**2))
                    )
                    * ck
                )

                stack[k, t_idx] = zz

        return stack

    def cp_bout_to_heatmap(self, bout, W, H, xv_rs, yv_rs):
        C, T, V, M = bout.shape
        zx = ((xv_rs) - bout[0, :, :, 0].ravel()) ** 2
        zy = ((yv_rs) - bout[1, :, :, 0].ravel()) ** 2
        zz = np.exp(-(zx + zy) / (2 * (0.1**2))) * bout[2, :, :, 0].ravel()
        zz = cp.swapaxes(cp.swapaxes(zz.reshape((W, H, T, V)), 0, -1), 1, 2)

        return zz

    def get_class_weights(self):
        # get class weights
        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(np.sort(self.labels)),
            y=self.labels,
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return class_weights

    def get_sample_weights(self):
        # get class weights
        class_weights = self.get_class_weights()
        class_weight_map = {k:v for k,v in enumerate(class_weights)}
        print("Class weight map is", class_weight_map)
        sample_weights = torch.tensor(pd.Series(self.labels).map(class_weight_map).to_numpy(dtype= "float32"), dtype=torch.float32)
        
        return sample_weights


class HyperParams:
    def __init__(self, batch_size, lr, dropout):
        self.batch_size = batch_size
        self.learning_rate = lr
        self.dropout = dropout
