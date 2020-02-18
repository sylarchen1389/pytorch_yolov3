from __future__ import division
 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors

def parse_cfg(cfgfile):
    file = open(cfgfile,'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x)>0]
    lines = [x for x in lines if x[0]! = '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == '[':
            if len(block) != 0 :
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)
    
    return blocks   

def create_model(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3 
    output_filters = []

    for index, block  in enumerate(blocks[1:]):
        module = nn.Sequential()
        # convolution layer
        if(block["type"] == "convolution"):
            activation = block["activation"]
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(block["filters"])
            padding = int(block["pad"]) 
            kernel_size = int(block["size"])
            stride = int(block["stride "])

            if padding: 
                pad = (kernel_size -1 )//2
            else:
                pad = 0
            
            conv = nn.Conv2d(prev_filters,filters,kernel_size,stride,pad,bias= bias)
            module.add_module("conv_{0}".format(index),conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index),bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1,inplace= True)
                module.add_module("leaky_{0}".format(index),activn)

        # upsample layer  
        elif(block["type"] == "'upsample"):
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=2,mode = "bilinear ")
            module.add_module("upsample_{0}".format(index),upsample)
        
        # route layer
        elif(block["type"] == "route"):
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])
            try:
                end = int(block["layers"][1])
            except:
                end = 0
            
            # positive anotation
            if start > 0:
                start = start - index 
            if end >0
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index),route)

            if end<0:
                filters = output_filters[index + start] + output_filter[index+end]
            else:
                filters = output_filters[index+start]
        
        # short cut 
        elif(block["type"]== "shortcut "):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index),shortcut)
        
        # yolo layer
        elif(block["type"] == "yolo"):
            mask = block["mask"].split(",")
            mask = [int(i) for i in mask ]

            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i],anchors[i+1])  for i in range(0,len(anchors),2) ]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index),detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info,module_list)

        