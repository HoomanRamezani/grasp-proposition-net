# Description 
This repository provides code and describes a deep learning grasp-proposition network designed to propose optimal grasp points for objects near a robotic arm. It contains a custom implementation of [PointNet (http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf) and [PointNet++](http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) in PyTorch. Users can utilize this repo to output grasp points on everyday objects, after feeding in PointNet data from a LiDAR camera.

## Preprocessing 
Download 3D indoor parsing dataset (**S3DIS**) [here](http://buildingparser.stanford.edu/dataset.html)  and save in `data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`.
```
cd data_utils
python collect_indoor3d_data.py
```
Processed data will save in `data/s3dis/stanford_indoor3d/`.

## Visualization
## build C++ code for visualization
cd visualizer
bash build.sh 
## run one example 
python show3d_balls.py
```
![](/visualizer/pic.png)

## Results
I started by training MedSam on a subset of the LIDC-IDRI dataset. This subset of data only included tumours larger than 14mm which resulted in a dataset of 550 lung slices. After performing 5 fold cross validation, I found that the model performs with an average of 0.893 dice coefficient. These results seem quite promising for my next step which is to train MedSAM on the full lidcidri dataset for tumors larger or equal to 3mm which represents about 10 500 lung images. 

However, these preliminary results where obtained by training the model only on lung slices that contained tumours. It is important to note that I ultimately want the model to take in as input the entire 3D lung scan which will inevitably also include lung slices that do not contain any tumors. Next steps are detailed below.

## Next Steps
### Further preprocessing
- As mentioned above, the goal is that the model performs well both on lung slices that contain and do not contain tumors. I am working on balancing the dataset to contain 60 % of the paired lung slices and annotations with no tumours and 40 % to contain tumours.
- Additionally, filtering 'closed' lung images which reside at the beginning and the end of the slice where the lung begins to close will enhance the efficiency of the model.
  
### Widening the scope
I am currently working on training MedSAM on the full lidcidri dataset with tumors larger or equal to 3mm which represents about 10 500 lung images.
'''
# Running the code
python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg
python test_classification.py --log_dir pointnet2_cls_ssg
'''

## Getting Started
- Download the [model checkpoint](https://drive.google.com/file/d/1tKd7p3cLVzvF3B4fpopijwNo2LSbKNWV/view?usp=drive_link) and place it in ```work_dir/SAM/```
- Download a [subset of the LIDC-IDRI dataset](https://drive.google.com/drive/folders/12xe3wfyHwUTbCv7XK-RoN1sRVuUXDZ1J?usp=sharing) and place it in ```MergedImages```

To start, run the script ```CentralScript_g.py```. This will run the model on a sample of the LIDC-IDRI dataset included in this repository.
