import json
import os
import matplotlib.pyplot as plt
import argparse


def main(args):
    f_json = open(args.work_dir, 'r')
    historys = f_json.readlines()

    bbox_mAP = []
    bbox_mAP_50 = []
    bbox_mAP_75 = []
    bbox_mAP_s = []
    bbox_mAP_m = []
    bbox_mAP_l = []

    segm_mAP = []
    segm_mAP_50 = []
    segm_mAP_75 = []
    segm_mAP_s = []
    segm_mAP_m = []
    segm_mAP_l = []

    for line in historys:
        print(line)
        json_ele = json.loads(line)

        if 'bbox_mAP' not in json_ele.keys():
            continue

        bbox_mAP.append(json_ele['bbox_mAP'])
        bbox_mAP_50.append(json_ele['bbox_mAP_50'])
        bbox_mAP_75.append(json_ele['bbox_mAP_75'])
        bbox_mAP_s.append(json_ele['bbox_mAP_s'])
        bbox_mAP_m.append(json_ele['bbox_mAP_m'])
        bbox_mAP_l.append(json_ele['bbox_mAP_l'])

        segm_mAP.append(json_ele['segm_mAP'])
        segm_mAP_50.append(json_ele['segm_mAP_50'])
        segm_mAP_75.append(json_ele['segm_mAP_75'])
        segm_mAP_s.append(json_ele['segm_mAP_s'])
        segm_mAP_m.append(json_ele['segm_mAP_m'])
        segm_mAP_l.append(json_ele['segm_mAP_l'])

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(bbox_mAP, label='bbox_mAP')
    ax[0].plot(bbox_mAP_50, label='bbox_mAP_50')
    ax[0].plot(bbox_mAP_75, label='bbox_mAP_75')
    ax[0].plot(bbox_mAP_s, label='bbox_mAP_s')
    ax[0].plot(bbox_mAP_m, label='bbox_mAP_m')
    ax[0].set_title('Bbox result')
    ax[0].legend()

    ax[1].plot(segm_mAP, label='segm_mAP')
    ax[1].plot(segm_mAP_50, label='segm_mAP_50')
    ax[1].plot(segm_mAP_75, label='segm_mAP_75')
    ax[1].plot(segm_mAP_s, label='segm_mAP_s')
    ax[1].plot(segm_mAP_m, label='segm_mAP_m')
    ax[1].set_title('Segm result')
    ax[1].legend()

    plt.ylim((0, 1))
    plt.show()
    fig.savefig('history.png')

    print('histroy figure saved.')

    # fname = str(args.work_dir).split('/')[-1].split('.')[0]
    # plt.savefig('{}.png'.format(fname))
    # print('{}.png saved.'.format(fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', type=str)
    args = parser.parse_args()

    main(args)
