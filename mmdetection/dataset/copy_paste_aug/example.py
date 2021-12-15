import cv2
import glob
import numpy as np
from copy_paste import CopyPaste
from coco import CocoDetectionCP
from visualize import display_instances
import albumentations as A
from skimage.filters import gaussian
import random
from matplotlib import pyplot as plt


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[..., None]
        img = paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)

    return img

transform = A.Compose([
        # A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
        A.PadIfNeeded(1000, 1000, border_mode=0), #pads with image in the center, not the top left like the paper
        A.RandomCrop(1000, 1000),
        # CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1.) #pct_objects_paste is a guess
    ], bbox_params=A.BboxParams(format="coco", min_visibility=0.05)
)

data = CocoDetectionCP(
    '../', 
    '../train_annotation.json', 
    transform
)

# f, ax = plt.subplots(1, 2, figsize=(16, 16))

past_paths = glob.glob('bg/*.jpg')
past_imgs = []
for img_path in past_paths:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_AREA)
    past_imgs.append(img)

copy_paste = CopyPaste(blend=True, sigma=1, pct_objects_paste=0.8, p=1.)
alpha = np.zeros((1000, 1000))
alpha[:,:] = 0.2

# index = random.randint(0, len(data))
for ele in data.coco.dataset["images"]:
    img_data = data[ele["id"] - 1]
    image = img_data['image']
    masks = img_data['masks']
    bboxes = img_data['bboxes']

    # rand_img = past_imgs[random.randint(0, len(past_imgs) - 1)]
    # for i in range(len(past_imgs)):
    #     f_name = ele["file_name"].split('/')[-1].split('.')[0]

    #     blend = cv2.cvtColor(past_imgs[i], cv2.COLOR_BGR2RGB)
    #     image = image_copy_paste(image, blend, alpha)
    #     image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    #     cv2.imwrite('../train/{}/images/{}_aug{}.png'.format(f_name, f_name, i + 1), image)
    #     print('../train/{}/images/{}_aug{}.png saved.'.format(f_name, f_name, i + 1))

    empty = np.array([])

    if len(bboxes) > 0:
        boxes = np.stack([b[:4] for b in bboxes], axis=0)
        box_classes = np.array([b[-2] for b in bboxes])
        mask_indices = np.array([b[-1] for b in bboxes])
        show_masks = np.stack(masks, axis=-1)[..., mask_indices]
        class_names = {k: data.coco.cats[k]['name'] for k in data.coco.cats.keys()}

        # display_instances(image, boxes, show_masks, box_classes, class_names, show_bbox=True, ax=ax[1])
        display_instances(image, boxes, show_masks, box_classes, class_names, show_bbox=True, img_id=ele["file_name"].split('/')[-1].split('.')[0])
    # else:
    #     display_instances(image, empty, empty, empty, empty, show_mask=False, show_bbox=False, img_id=ele["file_name"]split('/')[-1].split('.')[0])
        # display_instances(image, empty, empty, empty, empty, show_mask=False, show_bbox=False, ax=ax[1])