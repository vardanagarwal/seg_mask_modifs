# seg_mask_modifs

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fvardanagarwal%2Fmask_modifs&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## Description
A package for easy generation of mask of different labels using multiple models and applying different operations on that.

#### Curent models and labels supported:
- Deeplabv3 with pascal labels
- Maskrcnn with coco labels
- Bisnet with face labels

## Usage

### Installation
pip:
```
pip install seg-mask-modifs
pip install opencv-contrib-python>=4.5.4.60
# if you install opencv-python then inpainting won't work
```

Cloning repo then install requirements:
```
pip install -r requirements.txt
```

### Documentation

The documentation of the different classes and functions is available [here](https://vardanagarwal.github.io/seg_mask_modifs.html)

### Usage

### Download models

The models can be downloaded seperately or all of then can be downloaded at once.

```
from seg_mask_modifs import download_models

download_models.download_all() # download all models with default names which is highly recommended.

download_models.maskrcnn_coco(save_path='models/maskrcnn_restnet50_fpn.pt') # download maskrcnn model with coco labels
```

[Documentation page](https://vardanagarwal.github.io/seg_mask_modifs/download_models.html)

### Labels

To see the list of labels supported by the package, this function can be used.

```
from seg_mask_modifs import print_labels

print_labels.all() # prints all labels
print_labels.deeplab_pascal() # prints pascal labels
print_labels.maskrcnn_coco() # prints coco labels
print_label.face() # prints face labels
```
[Documentation page](https://vardanagarwal.github.io/seg_mask_modifs/print_labels.html)

### Mask Generation

Class to generate binary mask for any combination of labels. The models will be automatically used according to model preference and labels provided.

```
import cv2
from seg_mask_modifs import mask_generator

mask_gen = mask_generator.mask_generator(threshold=0.5, auto_init=True) # auto_init will only work if the models are saved to the default path.

# if auto_init is false or different path used to save model initialize them manually.
mask_gen.init_maskrcnn('maskrcnn.pt')
mask_gen.init_deeplab('deeplab.pt')
mask_gen.init_face('face.pth')

img = cv2.imread('images/city.jpg')
mask = mask_gen.generate(img=img, labels=['person', 'suitcase', 'hat'])
```

[Documentation page](https://vardanagarwal.github.io/seg_mask_modifs/mask_generator.html)

## References
1. Face parsing: https://github.com/zllrunning/face-parsing.PyTorch
2. Mask-RCNN: https://pytorch.org/vision/stable/_modules/torchvision/models/detection/mask_rcnn.html
3. Deeplabv3: https://pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html
