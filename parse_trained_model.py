import torch
# import torchvision
import collections
import os
from PIL import Image
import cv2
from torch.utils import data

# HOME = os.environ['HOME']

def parse_model(model_path):
    model_path = "./models/swin_large_patch4_window12_384_22kto1k.pth"

    assert(os.path.exists(model_path))

    x = torch.load(model_path)

    val = collections.OrderedDict()

    for key in x['model'].keys():
        # print(key, replace[key])
        print(key)
        val = x['model'][key]
        print(val.shape)

def parse_dataset(data_path, ann_dir):
    for img_path in os.listdir(data_path):
        im_path = os.path.join(data_path, img_path)
        ann_path = os.path.join(ann_dir, img_path)

        im = None
        # if data_type == 'anno':
        ann_im = Image.open(ann_path)
        # print(ann_im.size)
        try:
            assert(ann_im.mode == 'L')
        except Exception as ex:
            print('converting' , ann_path)
            ann_im = ann_im.convert('L')
            ann_im.save(ann_path)
        

        # else:
        im = cv2.imread(im_path)
        im_shape = (im.shape[1], im.shape[0])
        # print(im.shape)
        assert(im.shape[2] ==3)

        assert(im_shape == ann_im.size)

        

if __name__ == '__main__':
    data_path = './data/ade/ADEChallengeData2016/images/training'
    ann_path = './data/ade/ADEChallengeData2016/annotations/training'
    parse_dataset(data_path, ann_path)

