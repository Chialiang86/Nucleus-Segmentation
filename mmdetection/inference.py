from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets.coco import CocoDataset
import mmcv
import argparse
import json
import os


def main(args):
    config_file = args.config
    checkpoint_file = args.weight
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    test_f = open('dataset/test_img_ids.json')
    test_json = json.load(test_f)

    for element in test_json:
        prefix = 'dataset/test'
        img_path = os.path.join(prefix, element['file_name'])
        print('reading {} ...'.format(img_path))
        result = inference_detector(model, img_path)
        print(len(result[0][0]), len(result[1][0]))
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='', type=str)
    parser.add_argument('weight', default='', type=str)
    args = parser.parse_args()

    main(args)
