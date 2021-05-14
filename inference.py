from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

config_file = 'configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py'
checkpoint_file = 'models/upernet_swin_tiny_patch4_window7_512x512.pth'

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')


img = 'demo/demo.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_segmentor(model, img)

model.show_result(img, result, out_file='result.jpg')

