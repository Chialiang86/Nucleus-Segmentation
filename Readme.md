
# VRDL-HW3 Nucleus Instance Segmentation

> In this assignment, I applied [MMDetaction](https://github.com/open-mmlab/mmdetection) to finish my work.
 
## Dependencies
- Python : 3.7 (conda cirtual environment)
- Operating System : Ubuntu 20.04
- CUDA Version: 11.4 
- Packages : 
    - 1. Create a conda virtual environment and activate it.
        ```shell
        conda create -n openmmlab python=3.7 -y
        conda activate openmmlab
        ```
    - 2. Install PyTorch and torchvision following the official instructions
        ```shell
        conda install pytorch torchvision -c pytorch
        ```
    - 3. Install MMDetection
        ```shell
        pip install openmim
        mim install mmdet
        ```

## Shell Script For Running Different Task 
- Script Name : `run.sh` in `mmdetection/`
- Task : train / val / test/ inference
- Run Script : `sh run.sh [function : train/val/test/inference] [depends on function, results id or inference weight name]`
    - Example : 
        - training : `sh run.sh train 01`
        - val : `sh run.sh val 01`
        - test : `sh run.sh test 01`
        - inference : `sh run.sh inference epoch_50.pth`
- The script in `run.sh`
```shell 
# check command line argements
if [ "$#" -eq 2 ]; then 
    # config="configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py"
    # config="configs/mask_rcnn/mask_rcnn_r101_fpn_1x_coco.py"
    # config="decoupled_solo_r50_fpn_3x_coco.py"
    # config="quick_config/solo_r50_fpn_3x_coco.py"
    # config="quick_config/mask_rcnn_r101_fpn_2x_coco.py"
    config="quick_config/mask_rcnn_r101_fpn_2x_coco.py"
    function=$1
    echo "function : $1"

    if [ $1 = "train" ]; then
        weight="results/$2/latest.pth"
        echo "working directory = results/$2"
        python3.7 tools/train.py $config --work-dir results/$2
    elif [ $1 = "val" ]; then
        weight="results/$2/latest.pth"
        echo "working directory = results/$2"
        python3.7 tools/val.py $config $weight --work-dir results/$2 --show --eval segm
    elif [ $1 = "test" ]; then
        weight="results/$2/latest.pth"
        echo "working directory = results/$2"
        python3.7 tools/test.py $config $weight --work-dir results/$2 --format-only --options "jsonfile_prefix=answer"
        python3.7 format_answer.py
        rm answer.bbox.json
        rm answer.segm.json
    elif [ $1 = "inference" ]; then
        inference=$2
        echo "inference weight : $inference"
        python3.7 tools/test.py $config $inference --work-dir 'inference/' --format-only --options "jsonfile_prefix=answer"
        python3.7 format_answer.py
        rm answer.bbox.json
        rm answer.segm.json
    else
        echo "[error] function : $1"
    fi
else
    echo "[error] command line argument number should be 2"
fi
```

## Training
1. Download pretrained weight (R-101-FPN with Lr schd 2x in https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn).
2. Move the pretrained weight (`mask_rcnn_r101_fpn_2x_coco_bbox_mAP-0.408__segm_mAP-0.366_20200505_071027-14b391c7.pth`) to `mmdetection/quick_config/` .
3. Run shell script : `sh run.sh train [result name]`, the training results and weights will be in `mmdetection/results/[result name]/`.
    - Example : 
        `sh run.sh train 01`

## Validation
- Run shell script : `sh run.sh val [result name]`, the predicrting result and predicted image on the latest weight named `latest.pth` will be automatically feeded to `tool/val.py`  will be shown .
 
    - Example : 
        `sh run.sh val 01`
    
![](https://i.imgur.com/NvBpPHA.jpg)

```shell 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.112
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.589
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.149
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.256
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.187
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.305
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.314
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = -1.000
 ```
 
 ## Tesing
 
- Run shell script : `sh run.sh test [result name]`, `tool/test.py` will feed the latest weight in `results/[result name]/latest.pth` to generate the submission file `answer.json`
    - Example : 
    `sh run.sh test 09`


## Generate the best result for testing set
1. Download the best weight : [epoch_50.pth](https://drive.google.com/file/d/1Do6nT2s0vNMDdzJH1xUFgiwDfLaC6my8/view?usp=sharing)
2. Move the `epoch_50.pth` to `mmdetection/`
3. Run shell script `sh run.sh inference epoch_50.pth`
4. The `answer.json` was the best predicting result
5. Testing result : mAP = 0.24525

## References
- MMDetection : https://github.com/open-mmlab/mmdetection


