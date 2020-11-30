# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:19:16 2019
"""

from NASNET_conv import *
from models import *
from utils import *
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transform
import os,sys
from PIL import Image
import numpy as np
import scipy.misc


if len(sys.argv) < 2:
    raise SyntaxError
data_set = sys.argv[1]
imgs_path = sys.argv[2]

if data_set == 'salicon':
    nas_name = "checkpoint/coarse_salicon.pth"
    model_name = "checkpoint/fine_salicon.pth"

elif data_set == 'mit':
    nas_name = "checkpoint/coarse_mit.pth"
    model_name = "checkpoint/fine_mit.pth"


def make_hook(name):
    def hook(m, input, output):
        inter_feature[name] = output
    return hook

#Import parameters: coarse perceiving network (CFN-Coarse) 
with torch.no_grad():#没有梯度
    nas_model = NASNetALarge()#加载nas模型
    nas_model.load_state_dict(torch.load(nas_name, map_location=lambda storage, loc: storage))
    nas_model = nas_model.cuda(0)
    nas_model.eval()

#Import parameters: fine perceiving network (CFN-Fine) 
model = EMLNet()
#print(model)
pretrained_dict = torch.load(model_name)
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model = model.cuda(0)
model.eval()


test_data_dir = imgs_path
test_dataset = testdataset(test_data_dir)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1,
)
for step, (batch_x, batch_y, batch_z, batch_path) in enumerate(test_loader):
    with torch.no_grad():
        input ,width,length, path = batch_x, batch_y, batch_z, batch_path
        input = Variable(input).cuda(0)
        inter_feature = {}
        nas_model.reduction_cell_0.register_forward_hook(make_hook('feature2'))
        nas_model.reduction_cell_1.register_forward_hook(make_hook('feature3'))
        nas_model.cell_17.register_forward_hook(make_hook('feature4'))
        nas_model.cell_stem_0.register_forward_hook(make_hook('feature1'))
        nas_model.eval()
        nas_output = nas_model(input)
        
        x1 = Variable(inter_feature['feature1']).cuda(0)
        x2 = Variable(inter_feature['feature2']).cuda(0)
        x3 = Variable(inter_feature['feature3']).cuda(0)
        x4 = Variable(inter_feature['feature4']).cuda(0)
        
        prediction = model(x1,x2,x3,x4)
        prediction = prediction*255/torch.max(prediction)
        out = prediction.cpu().data.squeeze().numpy().astype('uint8')
        if data_set == 'mit':
            out = postprocess_predictions(out, length , width)
        print(os.path.split(path[0])[1])
        scipy.misc.imsave('output/'+ os.path.split(path[0])[1],out)
        
        
        
        
        
        
        
        