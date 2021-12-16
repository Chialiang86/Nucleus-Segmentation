import json
import os
import argparse


def main(args):

    fname = args.answer
    f_json = open(fname, 'r')
    answer_dict = json.load(f_json)
    for element in answer_dict:
        element['category_id'] = 1

    f_json_ans = open('answer.json', 'w')
    json.dump(answer_dict, f_json_ans, indent=4)
    print('answer.json done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer', '-a', default='answer.segm.json', type=str)
    args = parser.parse_args()

    main(args)
