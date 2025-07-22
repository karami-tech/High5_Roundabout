import shutil
import cv2
import numpy as np
import os
import math
import logging
import zipfile

from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from pycocotools.coco import COCO
from random import shuffle

# Script to take a COCO 1.0 zip-file in json format and convert this into YOLOv8 GT format
# It also copies image file to right folder
# Files train.txt, val.txt and test.txt are created based on the distribution given (default train-val-test: 80-20-0)

# json-file as exported from CVAT
COCO_gts_in = [    
    r"C:\Users\MulderLisa\Documents\Projecten\rotondes\test_coco_to_yolo.zip"
]

# Determine how files are divided into train/test/valid sets; examples: "60-20-20", "80-10-10", "80-20-0", "90-10-0"
# leave empty if output does not need to be split
distribution = "80-10-10"

logfile = r"C:\Users\MulderLisa\Documents\Projecten\rotondes\CVAT_COCO_1.0_to_YOLOv8.log" 
                
def get_distribution_percentages(in_str):
    numbers = in_str.split('-')
    return int(numbers[0])/100, int(numbers[1])/100, int(numbers[2])/100

def convert_cvat_json(json_file, out_dir):
    save_dir = make_dirs(out_dir)

    fn = Path(save_dir) / 'labels'  # folder name

    # Import json
    coco=COCO(json_file)

    # Create image dict
    images = {'%g' % x['id']: x for x in coco.imgs.values()}

    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    for ann in coco.anns.values():
        imgToAnns[ann['image_id']].append(ann)

    # Write yaml file
    yf = Path(save_dir) / Path(json_file).stem  # folder name
    with open((yf).with_suffix('.yaml'), 'w') as yaml_file:
        if distribution:
            yaml_file.write('\n'.join(['path: /data','train: train.txt', 'val: valid.txt', 'test: test.txt', '', 'names:', '']))
        else:
            yaml_file.write('\n'.join(['path: /data','train: images/train', 'val: images/valid', 'test: images/test', '', 'names:', '']))

        for cat in coco.cats.values():
            yaml_file.write(f"  {cat['id'] - 1}: {cat['name']}\n")
                            
    # Write labels file
    for img_id, anns in tqdm(imgToAnns.items(), desc=f'Annotations {json_file}'):
        img = images['%g' % img_id]
        h, w, f = img['height'], img['width'], img['file_name']

        bboxes = []
        segments = []
        for ann in anns:
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(ann['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            # Category
            cls = ann['category_id'] - 1

            # Boundingbox
            box = [cls] + box.tolist()
                
            # Segmentation
            if ann['iscrowd']:                    
                maskedArr = coco.annToMask(ann)
                mask = [polygonFromMask(maskedArr)]

                s = [j for i in mask for j in i]  # all segments concatenated
                s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist() # normalize
                s = [cls] + s
            else:
                if ann['segmentation'] == []:
                    logging.warning(f"Annotation is empty for image id {img_id} ({img['file_name']})")
                    s = None
                else:
                    if len(ann['segmentation']) > 1:
                        s = merge_multi_segment(ann['segmentation'])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist() # normalize
                    else:
                        s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist() # normalize
                    s = [cls] + s

            # Only add box and segments if segment was valid
            if s:
                if s not in segments:
                    segments.append(s)
                
                if box not in bboxes:
                    bboxes.append(box)
                    
        # Write
        # with open((fn / f).with_suffix('.txt'), 'a') as file:

        # label_path = fn / f
        # label_path.parent.mkdir(parents=True, exist_ok=True)  # zorg dat submappen zoals 'Images/' bestaan
        # with open(label_path.with_suffix('.txt'), 'a') as file:

        pure_file_name = os.path.basename(f)  # verwijdert 'Images/' of andere subfolders
        label_path = fn / pure_file_name
        # print(f)
        # print(label_path)
        # print(label_path.with_suffix('.txt'))
        with open(label_path.with_suffix('.txt'), 'a') as file:


            for i in range(len(bboxes)):
                line = *(segments[i]),  # cls, box or segments
                file.write(('%g ' * len(line)).rstrip() % line + '\n')

def make_dirs(dir='new_dir/'):
    # Create folders
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / 'labels':
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir

def polygonFromMask(maskedArr):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0

    for contour in contours:
    # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    if valid_poly == 0:
        raise ValueError
    
    return segmentation 

def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance. 
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all 
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...], 
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def write_meta_files(in_json_images_folder, images_folder_prepend, out_ann_dir, distribution):
    # Write lists of train, val and test images to files
    logging.info('Write lists of train, val and test images to files')
    if distribution:
        file_names = list((Path(in_json_images_folder)).glob("*.png"))
        # file_names = [f'{images_folder_prepend}{x.name}' for x in file_names]
        # file_names = [str(Path(images_folder_prepend) / x.name) for x in file_names]
        file_names = [str(Path(out_ann_dir).name + '/images/' + x.name) for x in file_names]

        perc_train, perc_val, _ = get_distribution_percentages(distribution)

        shuffle(file_names)
        first_split_index = math.ceil(len(file_names)*perc_train)
        second_split_index = first_split_index + math.ceil(len(file_names)*perc_val)
        train_files, valid_files, test_files = np.split(file_names, [first_split_index, second_split_index])

        with open(os.path.join(out_ann_dir, 'train.txt'),'w') as train_text_file:
            train_text_file.write('\n'.join(train_files.tolist()))

        with open(os.path.join(out_ann_dir, 'valid.txt'),'w') as valid_text_file:
            valid_text_file.write('\n'.join(valid_files.tolist()))

        with open(os.path.join(out_ann_dir, 'test.txt'),'w') as test_text_file:
            test_text_file.write('\n'.join(test_files.tolist()))

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ], force=True
    )
    
    logging.info('Logging started')

    for COCO_gt in COCO_gts_in:

        with zipfile.ZipFile(COCO_gt, 'r') as zip_ref:
            tmp_folder = os.path.join(os.path.dirname(COCO_gt), os.path.basename(COCO_gt).replace('.zip', '_tmp'))
            zip_ref.extractall(tmp_folder)

            # for in_folder in in_folders:
            in_folder = tmp_folder
            
            in_json_file = os.path.join(in_folder, r"annotations\instances_default.json")
            # in_json_images_folder = os.path.join(in_folder, r"images")
            in_json_images_folder = os.path.join(in_folder, r"images\Images")
            # out_ann_dir = in_folder + "_out"
            out_ann_dir = os.path.join(os.path.dirname(COCO_gt), os.path.basename(COCO_gt).replace('.zip', '_out'))

            # Convert COCO json file from CVAT to YOLOv8
            logging.info(f"Start processing '{in_json_file}'")
            convert_cvat_json(json_file = in_json_file,  out_dir = out_ann_dir)

            # Copy image files
            logging.info('Copy image files')
            shutil.copytree(in_json_images_folder, os.path.join(out_ann_dir, 'images'))

            write_meta_files(in_json_images_folder, 'images', out_ann_dir, distribution)

            shutil.rmtree(tmp_folder)

    logging.info('Logging gereed')

if __name__ == '__main__':
    main()