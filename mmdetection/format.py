import os
import json
import glob
import numpy as np
from PIL import Image
from pycocotools import mask
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import cv2
import argparse


def segs_area_bbox(img):
    # encoded_ground_truth = mask.encode(img)
    # ground_truth_area = mask.area(encoded_ground_truth)
    # ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(img, 0.5, positive_orientation='low')

    polygons = []
    segs = []
    for contour in contours:
        for i in range(len(contour)):
            row, col = contour[i]

            contour[i] = (
                col -
                1 if col -
                1 >= 0 else 0,
                row -
                1 if row -
                1 >= 0 else 0)  # from (row, col) to (x, y)

        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        if(poly.is_empty):
            # Go to next iteration, dont save empty values in list
            continue

        polygons.append(poly)

        seg = np.array(poly.exterior.coords).ravel().tolist()
        segs.append(seg)

    min_x, min_y, max_x, max_y = polygons[-1].bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygons[-1].area

    return segs, area, bbox


def main(args):
    training_dirs = os.listdir('dataset/train')
    testing_dirs = os.listdir('dataset/test')

    images = []
    annotations_train = []
    annotations_val = []
    categories = [
        {'id': 1, 'name': 'nucleus', 'supercategory': 'nucleus'},
    ]

    cnt = 0
    anno_id = 0
    for training_dir in training_dirs:
        print('processing : {}/{}'.format(cnt, len(training_dirs)))

        img_dir = os.path.join('dataset/train', training_dir, 'images')
        mask_dir = os.path.join('dataset/train', training_dir, 'masks')

        mask_paths = glob.glob('{}/*.png'.format(mask_dir))
        img_paths = glob.glob('{}/*.png'.format(img_dir))

        for img_path in img_paths:
            cnt += 1

            images.append({
                "file_name": img_path[8:],
                'height': 1000,
                'width': 1000,
                'id': cnt
            })

            for mask_path in mask_paths:
                img = Image.open(mask_path)
                img = np.asfortranarray(img) // 255

                segs, area, bbox = segs_area_bbox(img)

                if cnt > args.split_num:
                    annotations_train.append({
                        "segmentation": segs,
                        "area": area,
                        "iscrowd": 0,
                        "image_id": cnt,
                        "bbox": bbox,
                        "category_id": 1,
                        "id": anno_id
                    })
                else:
                    annotations_val.append({
                        "segmentation": segs,
                        "area": area,
                        "iscrowd": 0,
                        "image_id": cnt,
                        "bbox": bbox,
                        "category_id": 1,
                        "id": anno_id
                    })

                anno_id += 1

    coco_format_train = {
        "images": images[args.split_num:],
        "categories": categories,
        "annotations": annotations_train
    }

    coco_format_val = {
        "images": images[:args.split_num],
        "categories": categories,
        "annotations": annotations_val
    }

    with open('{}/train_annotation.json'.format(args.dir), 'w') as f:
        json.dump(coco_format_train, f, indent=4)

    with open('{}/val_annotation.json'.format(args.dir), 'w') as f:
        json.dump(coco_format_val, f, indent=4)

    # c = [628.9980392156863,629.0,628.0,628.0019607843137,627.9980392156863,628.0,627.9980392156863,627.0,627.0,626.0019607843137,626.9980392156863,626.0,626.0,625.0019607843137,625.9980392156863,625.0,625.0,624.0019607843137,624.9980392156863,624.0,624.0,623.0019607843137,623.0,623.0019607843137,622.0,623.0019607843137,621.9980392156863,623.0,621.0,622.0019607843137,620.0,622.0019607843137,619.9980392156863,622.0,619.0,621.0019607843137,618.0,621.0019607843137,617.0,621.0019607843137,616.0,621.0019607843137,615.0,621.0019607843137,614.0,621.0019607843137,613.0,621.0019607843137,612.0019607843137,622.0,612.0,622.0019607843137,611.0019607843137,623.0,611.0,623.0019607843137,610.0,623.0019607843137,609.0,623.0019607843137,608.0019607843137,624.0,608.0,624.0019607843137,607.0,624.0019607843137,606.0019607843137,625.0,606.0,625.0019607843137,605.0019607843137,626.0,605.0,626.0019607843137,604.0,626.0019607843137,603.0,626.0019607843137,602.0019607843137,627.0,602.0019607843137,628.0,602.0,628.0019607843137,601.0019607843137,629.0,601.0019607843137,630.0,601.0019607843137,631.0,601.0019607843137,632.0,601.0019607843137,633.0,602.0,633.9980392156863,602.0019607843137,634.0,602.0019607843137,635.0,603.0,635.9980392156863,603.0019607843137,636.0,604.0,636.9980392156863,605.0,636.9980392156863,605.0019607843137,637.0,606.0,637.9980392156863,606.0019607843137,638.0,607.0,638.9980392156863,608.0,638.9980392156863,608.0019607843137,639.0,609.0,639.9980392156863,610.0,639.9980392156863,611.0,639.9980392156863,612.0,639.9980392156863,613.0,639.9980392156863,614.0,639.9980392156863,615.0,639.9980392156863,616.0,639.9980392156863,616.9980392156863,639.0,617.0,638.9980392156863,618.0,638.9980392156863,618.9980392156863,638.0,619.0,637.9980392156863,620.0,637.9980392156863,621.0,637.9980392156863,621.9980392156863,637.0,622.0,636.9980392156863,622.9980392156863,636.0,623.0,635.9980392156863,623.9980392156863,635.0,624.0,634.9980392156863,624.9980392156863,634.0,625.0,633.9980392156863,625.9980392156863,633.0,626.0,632.9980392156863,626.9980392156863,632.0,627.0,631.9980392156863,627.9980392156863,631.0,627.9980392156863,630.0,628.0,629.9980392156863,628.9980392156863,629.0]
    # for i in range(0, len(c) - 2, 4):
    #     x1 = int(c[i])
    #     y1 = int(c[i+1])
    #     x2 = int(c[i+2])
    #     y2 = int(c[i+3])
    #     cv2.line(img, (y1, x1), (y2, x2), (0, 0, 255), 2)

    # cv2.imshow('s',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_num', '-s', default=5, type=int)
    parser.add_argument('--dir', '-d', default='dataset', type=str)
    args = parser.parse_args()

    main(args)
