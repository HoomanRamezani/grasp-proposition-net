import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement


class KITDataset(Dataset):
    def __init__(self, split='train',
                 data_root='/home/h2ramezani/Grasp/hooman/GraspDetection/Pointnet_Pointnet2_pytorch/data/kit',
                 num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        # define parameters
        super().__init__()
        self.num_point = num_point
        self.npoints = num_point

        self.block_size = block_size
        self.transform = transform
        self.sample_rate = sample_rate
        self.split = split

        self.root = data_root
        self.seg_classes = {}
        self.num_views = 3
        self.num_seg_classes = 52
        self.datapath = []

        # TO DO: make into parameter
        self.train_scene_numbers = np.arange(10)
        self.test_scene_numbers = np.arange(10, 13)

        self.scene_points, self.scene_labels = [], []
        self.scene_coord_min, self.scene_coord_max = [], []
        self.data_num = 0

        self.labelweights = np.zeros(52)
        self.labelweightstmp = np.zeros(52)

        self.num_point_all = []
        if self.split == 'train':
            self.len = len(self.train_scene_numbers)
            self.scene_numbers = self.train_scene_numbers
        else:
            self.len = len(self.test_scene_numbers)
            self.scene_numbers = self.test_scene_numbers

        # Get normalizing values, loop through all data and save
        data_num = 0
        for i in self.scene_numbers:
            scene_path = os.path.join(self.root, 'scene' + str(i))
            for j in range(self.num_views):
                prefix = os.path.join(scene_path, str(j) + '_')
                pcl = prefix + 'pcl.ply'
                ilabel = prefix + 'ilabel.txt'
                slabel = prefix + 'slabel.txt'

                fns = [pcl, slabel, ilabel]
                self.datapath.append(fns)

                semantic_seg = np.loadtxt(fns[1]).astype(np.int64)
                labels = semantic_seg

                plydata = PlyData.read(fns[0])
                # print(point_set.shape, seg.shape)
                x = plydata['vertex']['x']
                y = plydata['vertex']['y']
                z = plydata['vertex']['z']
                r = plydata['vertex']['red']
                g = plydata['vertex']['green']
                b = plydata['vertex']['blue']

                r = np.asarray(r).astype(np.float)
                g = np.asarray(g).astype(np.float)
                b = np.asarray(b).astype(np.float)

                points = np.stack((x, y, z, r, g, b), axis=1)

                tmp, _ = np.histogram(labels, range(53))
                coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
                self.scene_coord_min.append(coord_min)
                self.scene_coord_max.append(coord_max)
                data_num += 1

        # print(self.scene_coord_min)
        # print(len(self.scene_coord_min))
        # print(self.scene_coord_max)
        # print(len(self.scene_coord_max))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        """ GET DATA """
        scene_num = idx
        scene_points, scene_labels = [], []

        for j in range(self.num_views):
            view_num = j

            # Get paths
            scene_path = os.path.join(self.root, 'scene' + str(scene_num))
            prefix = os.path.join(scene_path, str(view_num) + '_')
            pcl = prefix + 'pcl.ply'
            ilabel = prefix + 'ilabel.txt'
            slabel = prefix + 'slabel.txt'
            fns = [pcl, slabel, ilabel]
            self.datapath.append(fns)

            # Load data
            print("loading data", self.data_num)
            semantic_seg = np.loadtxt(fns[1]).astype(np.int64)
            labels = semantic_seg
            bad_samples_array = []
            try:
                plydata = PlyData.read(fns[0])
                # print(point_set.shape, seg.shape)
                x = plydata['vertex']['x']
                y = plydata['vertex']['y']
                z = plydata['vertex']['z']
                r = plydata['vertex']['red']
                g = plydata['vertex']['green']
                b = plydata['vertex']['blue']
                r = np.asarray(r).astype(np.float)
                g = np.asarray(g).astype(np.float)
                b = np.asarray(b).astype(np.float)
                points = np.stack((x, y, z, r, g, b), axis=1)  # consolidate data
                tmp, _ = np.histogram(labels, range(53))
                self.labelweights += tmp
                coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
                self.scene_coord_min.append(coord_min)
                self.scene_coord_max.append(coord_max)
                self.num_point_all.append(labels.size)
                self.data_num += 1
            except OSError as e:
                bad_samples_array.append(prefix)
                print(">>> skip ", bad_samples_array, e.errno, fns[0])

            # Recalculate weights each time
            # Get label weights, can it be done individually or at the end
            labelweights = self.labelweights.astype(np.float32)
            labelweights = labelweights / np.sum(labelweights)
            temp_weights = labelweights / np.amax(labelweights)
            temp_weights[np.where(temp_weights == 0.)] = 1.
            temp_weights = np.power(temp_weights, -1 / 3.)
            self.labelweights = temp_weights

            # Don't need
            # sample_prob = self.num_point_all / np.sum(self.num_point_all)
            # num_iter = int(np.sum(self.num_point_all) * self.sample_rate / self.num_point)

            scene_points.append(points)
            scene_labels.append(labels)

        """ PROCESS DATA """
        # FIX
        points = np.array(scene_points)
        labels = np.array(scene_labels)

        # points = self.room_points[room_idx]   # N * 6
        # labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]
        print("N POINTS IS ", N_points)

        while (True):
            print(points[np.random.choice(N_points)][:3])
            center = points[np.random.choice(N_points)][:3]
            print(center)
            print([self.block_size / 2.0, self.block_size / 2.0, 0])
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                            points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.scene_coord_max[idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.scene_coord_max[idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.scene_coord_max[idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

        # Return data
        # data_idxs = []
        # if self.split == 'train':
        #     for index in range(self.data_num):
        #         data_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        # else:
        #     data_idxs = [ii for ii in range(self.data_num)]
        # self.data_idxs = np.array(data_idxs)
        # print("Totally {} samples in {} set.".format(len(self.data_idxs), self.data_num))


if __name__ == '__main__':
    # Not currently used
    print("KIT DATA LOADER")
