
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