import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement


class KITDataset(Dataset):
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area=5, block_size=1.0,
                 sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.npoints = num_point

        self.block_size = block_size
        self.transform = transform

        self.root = "/home/hooman.ramezani/grasp-detection/Pointnet_Pointnet2_pytorch/data/kit"
        self.seg_classes = {}
        self.num_views = 3

        self.num_seg_classes = 52

        self.datapath = []

        train_scene_numbers = np.arange(141)
        test_scene_numbers = np.arange(141, 152)

        if split == 'train':
            scene_numbers = train_scene_numbers
        else:
            scene_numbers = test_scene_numbers

        self.scene_points, self.scene_labels = [], []
        self.scene_coord_min, self.scene_coord_max = [], []

        labelweights = np.zeros(52)

        num_point_all = []

        data_num = 0
        for i in scene_numbers:
            scene_path = os.path.join(self.root, 'scene' + str(i))
            for j in range(self.num_views):
                prefix = os.path.join(scene_path, str(j) + '_')
                pcl = prefix + 'pcl.ply'
                ilabel = prefix + 'ilabel.txt'
                slabel = prefix + 'slabel.txt'

                fns = [pcl, slabel, ilabel]
                self.datapath.append(fns)

                print("loading data", data_num)
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
                labelweights += tmp
                coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
                self.scene_points.append(points)
                self.scene_labels.append(labels)
                self.scene_coord_min.append(coord_min)
                self.scene_coord_max.append(coord_max)
                num_point_all.append(labels.size)

                data_num += 1

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        temp_weights = labelweights / np.amax(labelweights)
        temp_weights[np.where(temp_weights == 0.)] = 1.
        temp_weights = np.power(temp_weights, -1 / 3.)

        self.labelweights = temp_weights

        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        data_idxs = []
        if split == 'train':
            for index in range(data_num):
                data_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        else:
            data_idxs = [ii for ii in range(data_num)]
        self.data_idxs = np.array(data_idxs)
        print("Totally {} samples in {} set.".format(len(self.data_idxs), data_num))

    def __getitem__(self, idx):

        data_idx = self.data_idxs[idx]
        points = self.scene_points[data_idx]
        labels = self.scene_labels[data_idx]

        # points = self.room_points[room_idx]   # N * 6
        # labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
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
        current_points[:, 6] = selected_points[:, 0] / self.scene_coord_max[data_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.scene_coord_max[data_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.scene_coord_max[data_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.data_idxs)


if __name__ == '__main__':
    data_root = '/data/yxu/PointNonLocal/data/stanford_indoor3d/'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=data_root, num_point=num_point, test_area=test_area,
                              block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random

    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()