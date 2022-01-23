from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from data_utils.KITDataLoader import KITDataset
import matplotlib.pyplot as plt
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
NUM_CLASSES = 52
# showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--dataset', type=str, default='', help='dataset path')
parser.add_argument('--class_choice', type=str, default='', help='class choice')

experiment_dir = 'log/sem_seg/pointnet2_sem_seg/'

opt = parser.parse_args()
print(opt)

d = KITDataset(
    split='test')

model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
MODEL = importlib.import_module(model_name)
classifier = MODEL.get_model(NUM_CLASSES).cuda()
checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier = classifier.eval()

total_seen_class = [0 for _ in range(NUM_CLASSES)]
total_correct_class = [0 for _ in range(NUM_CLASSES)]
total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

for idx in range(len(d)):

    print(d.data_idxs[idx])
    print("model %d/%d" % (idx, len(d)))
    point, seg = d[idx]
    point_np = np.array(point)
    point = torch.from_numpy(point_np).float()

    seg = torch.from_numpy(np.array(seg))
    gt = seg.numpy()
    # display

    cmap = plt.cm.get_cmap("hsv", 20)
    cmap = np.array([cmap(i) for i in range(20)])[:, :3]
    gt_color = cmap[seg.numpy(), :]

    point = point.transpose(1, 0).contiguous()

    point = Variable(point.view(1, point.size()[0], point.size()[1])).cuda()
    pred, _ = classifier(point)
    pred_choice = pred.data.max(2)[1]
    pred_choice = pred_choice.flatten()
    pred_choice = pred_choice.cpu().data.numpy()
    print(pred_choice)
    print(np.unique(gt), np.unique(pred_choice))
    # print(pred_choice.size())

    # display

    pred_color = cmap[pred_choice, :]

    # print(pred_color.shape)
    showpoints(point_np[:, :3], gt_color, pred_color)

    # import pdb
    # pdb.set_trace()
    for l in range(NUM_CLASSES):
        total_seen_class[l] += np.sum(gt == l)
        total_correct_class[l] += np.sum((gt == l) & (pred_choice == l))
        total_iou_deno_class[l] += np.sum(((gt == l) | (pred_choice == l)))

IOU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
accuracy = np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)

print("IOU:", IOU, "mIOU", np.sum(IOU) / np.sum(IOU > 0.), "accuracy:", accuracy)
