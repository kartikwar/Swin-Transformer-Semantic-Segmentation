import torch
# import torchvision
import collections
import os

# HOME = os.environ['HOME']
model_path = "./models/swin_large_patch4_window12_384_22kto1k.pth"

assert(os.path.exists(model_path))

x = torch.load(model_path)

val = collections.OrderedDict()

for key in x['model'].keys():
    # print(key, replace[key])
    print(key)
    val = x['model'][key]
    print(val.shape)
    # val[key.replace('module', 'encoder')] = x[key]

# y = {}
# y['state_dict'] = val
# y['epoch'] = 0

# torch.save(y, './models/encoder.pth')
