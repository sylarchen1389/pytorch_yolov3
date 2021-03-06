from __future__ import division
 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 

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
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    count = 0
    for line in lines:
        count  = count +1
        if line[0] == '[':
            if len(block) != 0 :
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        elif line.rstrip() == "Training" or line.rstrip() == "Testing":
            block["Mode"] = line.rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    print("parse cfg file success")
    return blocks   

def create_model(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3 
    output_filters = []

    print(net_info)
    for index, block  in enumerate(blocks[1:]):
        module = nn.Sequential()
        # convolution layer
        print(index,":",block)
        if(block["type"] == "convolutional"):
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
            stride = int(block["stride"])

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
            if activation == "ReLU":
                module.add_module("ReLU_{}".format(index), nn.ReLU())

        # upsample layer  
        elif(block["type"] == "upsample"):
            upsample = nn.Upsample(scale_factor=int(block["stride"]),mode = "nearest")
            module.add_module("upsample_{0}".format(index),upsample)
        
        # route layer
        elif(block["type"] == "route"):
            block["layers"] =[ int(a) for a in block["layers"].split(',')]
            start = int(block["layers"][0])
            try:
                end = int(block["layers"][1])
            except:
                end = 0
            
            # positive anotation
            if start > 0:
                start = start - index
            if end >0: 
                end = end - index
            

            if end<0:
                filters = output_filters[index + start] + output_filters[index+end]
            else:
                filters = output_filters[index+start]
        
        # short cut 
        elif(block["type"]== "shortcut"):
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


class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info,self.module_list = create_model(self.blocks)
    
    def forward(self,x,CUDA):
        modules =self.blocks[1:]
        outputs = {}

        write = 0 
        for i,module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]>0):
                    layers[0] = layers[0] - i
                
                if(len(layers)==1):
                    x = outputs[i+(layers[0])]
                else:
                    if(layers[1]>0):
                        layers[1] = layers[1] - i

                    map1 = outputs[i+(layers[0])]
                    map2 = outputs[i+(layers[1])] 
                    x = torch.cat([map1,map2],1)
            
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]
            
            elif module_type == 'yolo':

                anchors = self.module_list[i][0].anchors
                inp_h,inp_w = int(self.net_info["height"]),int(self.net_info["width"])
                num_classes = int(module["classes"])

                #Transform 
                x = x.data
                x  = predict_transform(x,inp_w,inp_h,anchors,num_classes,CUDA= True)

                if not write:
                    detections = x 
                    write  =1 
                else:
                    detections = torch.cat((detections,x),1)
            outputs[i] = x 
        return detections
    
    def load_weights(self,weightfile):
        weight_fp = open(weightfile,"rb")
        header = np.fromfile(weight_fp,dtype = np.int32, count=5 )
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(weight_fp,dtype=np.float32)

        ptr =  0 
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try: 
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                if batch_normalize:
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()
                    
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+= num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_bias = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_bias])
                    ptr = ptr + num_bias                
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)



def get_test_input():
    img = cv2.imread('data/dog.jpg')
    img = cv2.resize(img,(608,608))
    img_ = img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_

if __name__ == "__main__":
    #blocks = parse_cfg('cfg/yolov3.cfg')
    #print(create_model(blocks))
    model = Darknet("cfg/yolov3.cfg")
    print("loading weights...")
    #model.load_weights("yolov3.weights")
    inp = get_test_input()
    pred = model(inp,torch.cuda.is_available())
    outputs = write_results(pred,0.5,80)
    print(outputs.shape)    
    