from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *

def get_test_input():
    img = cv2.imread('dog-cycle-car.png')
    img = cv2.resize(img, (416, 416))
    img_new = img[:,:,::-1]
    #::-1 means take everything in dim 2 but backwards, which is changing BGR to RGB.
    #OpenCV loads image with BGR as the order, PyTorch loads image with RGB as the order
    img_new = img_new.transpose((2,0,1))
    #img:(H, W, C) img_new:(C, H, W)
    img_new = img_new[np.newaxis,:,:,:]/255.0
    img_new = torch.from_numpy(img_new).float()
    img_new = Variable(img_new)
    return img_new

def parse_cfg(cfgfile): #Return a list of blocks, each element in the list are dictionaries, represent a block in NN
    file = open(cfgfile, 'r')
    lines = file.read().split('\n') #Put the content in each line into a list
    lines = [x for x in lines if len(x) > 0] #Delete the empty lines
    lines = [x for x in lines if x[0] != '#'] #Delete the comments
    lines = [x.rstrip().lstrip() for x in lines] #Delete the spaces in left and right

    blocks = []
    block = {}

    for line in lines:

        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].rstrip() #e.g. block['type']=convolutional
        else:
            key, value = line.split('=') #key='size', value=3
            block[key.rstrip()] = value.lstrip() #e.g. block['size'] = 3

    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
"""
The code of concatenation is short and simple, put concatenation in layer will lead
increase boilerplate code. So we put a dummy layer first, then perform the concatenation
in the foward function.
"""

#Define a detection layer to hold the anchors used to detect bounding boxes
class DetectionLayer(nn.Module):

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.MSELoss = nn.MSELoss()
        self.BCELoss = nn.BCELoss()

    def forward(self, prediction, input_dim, num_classes, targets, ignore_thres):
        """
        prediction: 4-D tensor (batch_size, box_para*num_anchors, grid_size, grid_size)
        Turns an detection feature map into a 3-D tensor, (batch_size, num_anchors*grid_size*grid_size, num_classes + 5)
        each row in each image corresponds to a bounding box:
        [[1st box at (0,0)],
         [2nd box at (0,0)],
         [3rd box at (0,0)],
         [1st box at (0,1)],
         ......
         [1st box at (5,0)],
         ......             ]
        each box has num_classes+5 parameters(tx, ty, tw, th, confidence + probability of num_classes classes)
        """
        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        batch_size = prediction.size(0)
        grid_size = prediction.size(2)
        stride = input_dim // grid_size
        box_para = 5 + num_classes
        num_anchors = len(self.anchors)
        #The sizes of anchors defined in cfg are with respect to the size of input image, we need to change the
        #size of anchors with respect to the size of feature map
        scaled_anchors = FloatTensor([(a[0]/stride, a[1]/stride) for a in self.anchors])
        anchor_w = scaled_anchors[:, 0].view((1, num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1].view((1, num_anchors, 1, 1))

        prediction = (
            prediction.view(batch_size, num_anchors, box_para, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2).contiguous())
        #prediction: (batch_size, num_anchors, grid_size, grid_size, num_classes + 5)
        #If no .contiguous(), cannot do .view()
        #Apply sigmoid function on t_x, t_y and confidence in each bounding box
        x = torch.sigmoid(prediction[...,0])
        y = torch.sigmoid(prediction[...,1])
        w = prediction[...,2]
        h = prediction[...,3]
        pred_conf = torch.sigmoid(prediction[...,4])
        pred_cls = torch.sigmoid((prediction[...,5:5+num_classes]))
        """
        e.g. grid_size = 5
        grid_x = [[[[0 1 ... 4]            grid_y = [[[[0 0 0 0 0]
                    [0 1 ... 4]                        [1 1 1 1 1]
                     .........                          .........
                    [0 1 ... 4]]]](5 rows)             [4 4 4 4 4]]]]
        """
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(FloatTensor)
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(FloatTensor)

        #pred_boxes: (batch_size, num_anchors, grid_size, grid_size, 4) the last dim is [x, y, w, h]
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        '''
        e.g. pred_boxes[..., 0]:
        [[image1, 1st box:
          [[x at (0,0), x at (0,1), x at (0,2), ... x at (0,4)],
           [x at (1,0), x at (1,1), x at (1,2), ... x at (1,4)],
           .......
           [x at (4,0), x at (4,1), x at (4,2), ... x at (4,4)]],
          image1, 2nd box:
           .......]]
        '''
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        #* stride: Resize the detection map to the size of input image
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * stride,
                            pred_conf.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, num_classes), ), -1,)

        if targets is None:
            return output, 0
        else:
            #Compute loss, pred_boxes: x, y, w, h
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                    pred_boxes=pred_boxes,
                    pred_cls=pred_cls,
                    target=targets,
                    anchors=scaled_anchors,
                    ignore_thres=ignore_thres,
            )

            #Mask outputs to ignore non-existing objects
            loss_x = self.MSELoss(x[obj_mask], tx[obj_mask])
            loss_y = self.MSELoss(y[obj_mask], ty[obj_mask])
            loss_w = self.MSELoss(w[obj_mask], tw[obj_mask])
            loss_h = self.MSELoss(h[obj_mask], th[obj_mask])
            loss_conf_obj = self.BCELoss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.BCELoss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = loss_conf_obj + 100 * loss_conf_noobj
            loss_cls = self.BCELoss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return output, total_loss

def create_modules(blocks):

    net_info = blocks[0] #The first block contains the hyperparameters of the network
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = [] #Number of channels at the end of each block

    for i, block in enumerate(blocks[1:]):

        modules = nn.Sequential() #A block may contain more than one layer

        if block['type'] == 'convolutional':

            activation = block['activation']
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            padding = int(block['pad'])

            try:
                batch_normalize = int(block['batch_normalize'])
                bias = False #Do not need bias in conv layer if we do BN
            except:
                batch_normalize = 0
                bias = True

            if padding:
                pad = (kernel_size - 1) // 2 #Make sure every pixel is the center of kernel
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            modules.add_module('conv_' + str(i), conv) #e.g. add module conv_1

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                modules.add_module('batch_norm_' + str(i), bn)

            if activation == 'leaky':
                act = nn.LeakyReLU(0.1, inplace = True)
                modules.add_module('leaky_' + str(i), act)

        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            ups = nn.Upsample(scale_factor = stride, mode = 'nearest')
            modules.add_module('upsample_' + str(i), ups)

        elif block['type'] == 'route':
            """
            In cfg file, route layer is defined as:
            [route]
            layers = -1, 36
            or
            [route]
            layers = -4

            If there is one value in "layers", route layer outputs the feature maps
            of the specified layer
            If there are two values in "layers", route layer outputs the concatenated
            feature maps of the specified layers
            """
            layers = block['layers'].split(',')
            start = int(layers[0])

            try:
                end = int(layers[1])
            except:
                end = 0

            if start < 0:
                start = i + start #start layer = (i+start)th layer when start<0

            if end < 0:
                end = i + end

            route = EmptyLayer()
            modules.add_module("route_" + str(i), route)

            if end == 0:
                filters = output_filters[start]
            else:
                filters = output_filters[start] + output_filters[end]

        elif block['type'] == 'shortcut':
            shortcut = EmptyLayer()
            modules.add_module('shortcut_' + str(i), shortcut)

        elif block['type'] == 'yolo':
            mask = block['mask'].split(',')
            mask = [int(m) for m in mask]

            anchors = block['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            #anchors are noted as anchors = 10,13,  16,30... in cfg
            anchors = [(anchors[n], anchors[n+1]) for n in range(0, len(anchors),2)]
            #anchors are the same in each block, use index of mask to choose which anchor to use
            anchors = [anchors[n] for n in mask]

            detection = DetectionLayer(anchors)
            modules.add_module('Detection_' + str(i), detection)

        module_list.append(modules)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)

class Darknet(nn.Module):

    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, targets = None):
        blocks = self.blocks[1 : ]
        modules = self.module_list
        outputs = {}
        #A dictionary contains output feature maps of every route and shortut layers.
        #key: index of layer, value: feature maps
        detected = 0 #detected=0 means have not made yolo detection yet
        loss = 0
        for i, (block, module) in enumerate(zip(blocks, modules)):

            if block['type'] == 'convolutional' or block['type'] == 'upsample':
                x = self.module_list[i](x)

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                start = int(layers[0])

                try:
                    end = int(layers[1])
                except:
                    end = 0

                if start < 0:
                    start = i + start #start layer = (i+start)th layer when start<0

                if end < 0:
                    end = i + end

                if end == 0:
                    x = outputs[start]
                else:
                    cat1 = outputs[start]
                    cat2 = outputs[end]
                    x = torch.cat((cat1, cat2), 1)

            elif block['type'] == 'shortcut':
                start = int(block['from'])
                x = outputs[i - 1] + outputs[i + start]

            elif block['type'] == 'yolo':
                input_dim = int(self.net_info['height'])
                num_classes = int(block['classes'])
                #x: (batch_size, box_para(5 + classes)*num_anchors, grid_size, grid_size)
                x, layer_loss = module[0](x, input_dim, num_classes, targets, ignore_thres = 0.5)
                loss = loss + layer_loss

                if detected == 0:
                    detections = x
                    detected = 1
                else:
                    detections = torch.cat((detections, x), 1)
                    #detections: (batch_size, grid_size*grid_size*num_anchors*3, 5+num_classes)

            outputs[i] = x
        if targets is None:
            return detections
        else:
            return loss, detections

    def load_weights(self, weightfile):
        print('weights loaded')
        fp = open(weightfile, 'rb')
        #The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype = np.float32) #Load the rest of fp

        v = 0 #Read the vth value in weights

        for i in range(len(self.module_list)): #Go through all the layers, find conv layer
            module_type = self.blocks[i + 1]['type']
            #If module_type is conv, load weights.
            if module_type == 'convolutional':
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]['batch_normalize'])
                except:
                    batch_normalize = 0

                conv = model[0] #The 1st layer in each conv module is conv

                if (batch_normalize):
                    bn = model[1] #The 2nd layer in each conv module is BN if there is a BN layer
                    #Get the number of channels of BN layer
                    num_bn_c = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[v:v + num_bn_c])
                    v += num_bn_c

                    bn_weights = torch.from_numpy(weights[v:v + num_bn_c])
                    v += num_bn_c

                    bn_running_mean = torch.from_numpy(weights[v:v + num_bn_c])
                    v += num_bn_c

                    bn_running_var = torch.from_numpy(weights[v:v + num_bn_c])
                    v += num_bn_c

                    #Reshape bn_bias to the same shape as bn.bias.data
                    #bn.bias and bn.weight are parameter object containing tensor, .data change them into just tensor
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    #bn.running_mean and bn.running_var are tensors, therefore no .data needed
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data from weights into model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv.bias.numel() #Get the number of elements in bias

                    conv_biases = torch.from_numpy(weights[v:v+num_biases])
                    v += num_biases

                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                #Load the weights of conv layers
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[v:v+num_weights])
                v += num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)










'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Darknet("cfg/yolov3.cfg")
model.load_weights('yolov3.weights')
model = model.to(device)
inp = get_test_input()
inp = inp.to(device)
pred = model(inp, torch.cuda.is_available())
print(pred)
'''
