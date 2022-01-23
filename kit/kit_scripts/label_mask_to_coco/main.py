import glob
import cv2
import os

from create_annotations import *

# Class name define
# Label ids of the dataset
category_ids = {
"Cheezit ": 0 ,
"Can ": 1 ,
"Banana ": 2 ,
"Redcup ": 3 ,
"Drill ": 4 ,
"Scissors ": 5 ,
"Strawberry ": 6 ,
"Peach ": 7 ,
"Pear ": 8 ,
"Purplefruit ": 9 ,
"Knife ": 10 ,
"Screwdriver ": 11 ,
"Blueball ": 12 ,
"Bluecup ": 13 ,
"Bluetoy2 ": 14 ,
"UnknownObject ": 15 ,
"Bluetoy ": 16 ,
"itm 18 ": 17 ,
"Smallbottle ": 18 ,
"Handsoup2 ": 19 ,
"Bottle1 ": 20 ,
"Bottle2 ": 21 ,
"Dishwasher ": 22 ,
"Toothpaste ": 23 ,
"Orangemarker ": 24 ,
"Handsoup ": 25 ,
"Bowl ": 26 ,
"ToyCamel ": 27 ,
"ToyElephant ": 28 ,
"ToyRhinoceros ": 29 ,
"ToyKingkong ": 30 ,
"Cup ": 31 ,
"Toothpastebox ": 32 ,
"Mouse ": 33 ,
"Bottle3 ": 34 ,
"Handcream ": 35 ,
"Handcream2 ": 36 ,
"Shampoo ": 37 ,
"Hippopotamus ": 38 ,
    "Tape ": 39 ,
    "sugarbox ":40,
    "spam ":41,
    "unlabeled1 ":42,
    "unlabeled2 ":43,
    "unlabeled3 ":44,
    "unlabeled4 ":45,
    "unlabeled5 ":46,
    "unlabeled6 ":47,
    "unlabeled7 ":48,
    "unlabeled8 ":49,
    "unlabeled9 ":50,
    "unlabeled10 ":51,
}

# Define which colors (in image) match which categories (Label ID defined above) in the images
category_colors = {
0 : 0 ,
1 : 40,
2 : 1 ,
    3:42,
    4:41,
5 : 2 ,
    6:43,
7 : 3 ,
8 : 4 ,
9 : 5 ,
    10:44,
11 : 6 ,
    12:45,
    13:46,
14 : 7 ,
15 : 8 ,
    16:47,
17 : 9 ,
18 : 10 ,
    19:48,
20 : 11 ,
21 : 12 ,
22 : 13 ,
    23:49,
    24:50,
    25:51,
26 : 14 ,
27 : 15 ,
29 : 16 ,
30 : 17 ,
34 : 18 ,
36 : 19 ,
37 : 20 ,
38 : 21 ,
40 : 22 ,
41 : 23 ,
43 : 24 ,
44 : 25 ,
46 : 26 ,
48 : 27 ,
51 : 28 ,
52 : 29 ,
56 : 30 ,
57 : 31 ,
58 : 32 ,
60 : 33 ,
61 : 34 ,
62 : 35 ,
63 : 36 ,
66 : 37 ,
69 : 38 ,
70 : 39 ,
}

selected_class = list(category_ids.keys())

selected_id = [category_ids[c] for c in selected_class]

# Define the ids that are a multiplolygon. For example: wall, roof and sky
# Currently listing all all of them as multipolygon as there are occlusions
multipolygon_ids = list(category_colors.keys())

# Get "images" and "annotations" info 
def images_annotations_info(path, start, end):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []

    #generate data for scene number from [start] to [end]
    for i in range(start, end+1):
        scene_num = str(i)
        scene_path = os.path.join(path, "scene"+scene_num)

        #going through all 9 images in the scene
        for j in range(9):
            #original file is the input for the network
            original_file_name = os.path.join(scene_path, str(j)+'_rgb.png')

            #parsing masks from .npz file provided by Max
            numpy_file = os.path.join(scene_path, str(j)+'.npz')
            data = np.load(numpy_file)
            instances_objects = data['instances_objects']
            instances_semantic = data['instances_semantic']
            mask_image_open = instances_semantic

            h = mask_image_open.shape[0]
            w = mask_image_open.shape[1]

            # create image (input) annotation
            image = create_image_annotation(original_file_name, w, h, image_id)
            images.append(image)

            # create masks annotation for each object instance, we will check the actual class label later
            sub_masks = create_sub_masks(instances_objects, w, h)
            #find all object instances
            for object_id, sub_mask in sub_masks.items():
                #find the class color of the instance in semantic mask
                color = instances_semantic[([np.where(sub_mask > 0)[0][0]], [np.where(sub_mask > 0)[1][0]])][0]
                color = int(color)
                category_id = category_colors[color]
                if not category_id in selected_id:
                    continue

                # "annotations" info
                polygons, segmentations = create_sub_mask_annotation(sub_mask)

                # Check if we have classes that are a multipolygon
                if category_id in multipolygon_ids:
                    # Combine the polygons to calculate the bounding box and area
                    multi_poly = MultiPolygon(polygons)
                    try:
                        min_x, min_y, max_x, max_y = multi_poly.bounds
                    except:
                        continue
                    annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id)

                    annotations.append(annotation)
                    annotation_id += 1
                else:
                    for i in range(len(polygons)):
                        # Cleaner to recalculate this variable
                        segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]

                        annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id)

                        annotations.append(annotation)
                        annotation_id += 1
            image_id += 1
    return images, annotations, annotation_id


if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()

    # define your dataset path
    pathes = {"train": "/home/hooman.ramezani/Desktop/Grasp/grasp-detection/Pointnet_Pointnet2_pytorch/data/kit-easy",
              "val": "/home/hooman.ramezani/Desktop/Grasp/grasp-detection/Pointnet_Pointnet2_pytorch/data/kit-easy"}
    keywords = ["train", "val"]
    for keyword in keywords:
        root_path = pathes[keyword]
        # Create category section
        coco_format["categories"] = create_category_annotation(category_ids)
        # Create images and annotations sections for each dataset, where we use all 2500 images for training, and 250 images for test/val
        if keyword == "train":
            coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(root_path, 0, 10)
        else:
            coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(root_path, 0, 2)
            
        with open("output/{}.json".format(keyword),"w") as outfile:
            json.dump(coco_format, outfile)

        print("Created %d annotations for images in folder: %s" % (annotation_cnt, root_path))

