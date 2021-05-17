from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'configs/swin/saliency_swin_tiny.py'
checkpoint_file = 'work_dirs/saliency_swin_tiny/iter_100.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')


img = 'data/ade/ADEChallengeData2016/images/training/20245.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)

model.show_result(img, result, out_file='result_20245.jpg')

