from __future__ import division
 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2

def predict_transform(prediction , inp_w , inp_h , anchors , num_classes, CUDA = True ):
    batch_size = prediction.size(0)
    nGh = prediction.size(2)
    nGw = prediction.size(3)

    stride = inp_h //nGh
    grid_size = inp_h // stride ## nGh?
    bbox_attrs = num_classes + 5
    num_anchors = len(anchors)

    # torch.Tensor.veiw, reshape
    prediction = prediction.view(batch_size,bbox_attrs*num_anchors,grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size,grid_size*grid_size*num_anchors,bbox_attrs)

    anchors= [(a[0]/stride,a[1]/stride) for a in anchors]

    prediction[: , : ,0] = torch.sigmoid(prediction[:,:,0]) #x
    prediction[: , : , 1] = torch.sigmoid(prediction[: , : , 1]) #y
    prediction[: , : ,4] =   torch.sigmoid(prediction[: , : ,4]) # conf

    # prepare anchors offsett 
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid,grid)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA: 
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset,y_offset),1).repeat(1,num_anchors).view(-1,2)

    prediction[:,:,:2] = x_y_offset

    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()
    
    # add dim 
    # (1,grid_size*grid_size*num_anchors,2) 
    anchors = anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors # wh 
    # class conf
    prediction[:,:,5:5+num_classes] = torch.sigmoid(prediction[:,:,5:5+num_classes])

    prediction[:,:,:4]  *= stride # xywh

    return prediction




