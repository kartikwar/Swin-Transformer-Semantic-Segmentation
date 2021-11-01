python train.py \
    configs/swin/saliency_swin_large.py \
    --options=model.pretrained=models/swin_large_patch4_window12_384_22kto1k.pth \
    --options=model.backbone.use_checkpoint=True\
    --no-validate
