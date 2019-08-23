from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

print('no train')

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def load_classes(namesfile):
    fp = open(namesfile, 'r')
    names = fp.read().split('\n')[:-1] #The last element in fp.read().split('\n') is ' '
    return names

#Resize the image with unchanged aspect ratio, padding left out areas with color (128, 128, 128)
def letterbox_image(img, inp_dim):
    img_w, img_h = img.shape[1], img.shape[0] #img_w and img_h are w and h of the input
    w, h = inp_dim #w, h are w, h of resized image
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_CUBIC) #Do cubic interpolation during resizing image

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128) #Create a (h, w, 3) array with every elements equal to 128
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) //2 + new_w,:] = resized_image

    return canvas

#Transport image to PyTorch input
def prep_image(img, inp_dim):
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2, 0, 1)).copy() #BGR in opencv to RGB in PyTorch, (H, W, C) to (C, H, W)
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0) #(C, H, W) to (1, C, H, W), the new dim represent batch
    return img

def predict_transform(prediction, input_dim, anchors, num_classes, CUDA):
    """
    prediction: 4-D tensor (batch_size, box_para*num_anchors, grid_size, grid_size)
    Turns an detection feature map into a 2-D tensor, each row corresponds to a bounding box:
    [[1st box at (0,0)],
     [2nd box at (0,0)],
     [3rd box at (0,0)],
     [1st box at (0,1)],
     ......
     [1st box at (5,0)],
     ......             ]
    each box has num_classes+5 parameters(tx, ty, tw, th, confidence + probability of num_classes classes)
    """
    batch_size = prediction.size(0)
    stride = input_dim // prediction.size(2)
    grid_size = input_dim // stride
    box_para = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(batch_size, box_para * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    #prediction: (batch_size, grid_size * grid_size, num_anchors * box_para)
    #If no .contiguous, cannot do .view()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, box_para)
    #The sizes of anchors defined in cfg are with respect to the size of input image, we need to change the
    #size of anchors with respect to the size of feature map
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    #Apply sigmoid function on t_x, t_y and confidence in each bounding box
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    """
    e.g. grid_size = 5, grid = [0, 1, 2, 3, 4]
    a = [[0 1 ... 4]            b = [[0 0 0 0 0]
         [0 1 ... 4]                 [1 1 1 1 1]
          .........                   .........
         [0 1 ... 4]](5 rows)        [4 4 4 4 4]]
    """
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)
    """
    x_offset = [[0],[1],[2],[3],[4],[0],[1],...]
    y_offset = [[0],[0],[0],[0],[0],[1],[1],...]
    """
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
    """
    e.g. num_anchors = 3
    x_y_offset = [[[0,0],[0,0],[0,0],[1,0],[1,0]...[4,4]]]
    the elements in x_y_offset are (cx, cy)
    """
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)
    """
    bx = sig(tx) + cx
    by = sig(ty) + cy
    """
    #prediction[:,:,:2] += x_y_offset
    prediction[:,:,:2] = prediction[:,:,:2] + x_y_offset

    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()
    #anchors = [[[anchor1],[anchor2],[anchor3],[anchor1],[anchor2],...(repeat grid_size*grid_size times)]
    #x.repeat(a,b): repeat x a times on dim 0, b times on dim 1
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    """
    bw = pw*e^tw
    bh = ph*e^th
    """
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    #Apply sigmoid activation to class scores
    prediction[:,:,5:5 + num_classes] = torch.sigmoid((prediction[:,:,5:5+num_classes]))
    #
    #Resize the detection map to the size of input image
    prediction[:,:,:4] *= stride

    return prediction

def bbox_iou(box1, box2):
    #box1, box2 are (1,7) and (n,7) tensors
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    #b1_x1... are (1,1) tensors
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    #b2_x1... are (n,1) tensors

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.max(b1_y2, b2_y2)
    #inter_x1... are (n,1) tensors

    inter_area = torch.clamp(inter_x2 + 1 - inter_x1, min = 0) * torch.clamp(inter_y2 +1 - inter_y1, min = 0)
    #Because we are dealing with pixels, the bottom-right coordinate of intersection is (inter_x2+1, inter_y2+1)
    #torch.clamp(x, min=0): if x < 0, return 0

    b1_area = (b1_x2 + 1 - b1_x1) * (b1_y2 + 1 - b1_y1)
    b2_area = (b2_x2 + 1 - b2_x1) * (b2_y2 + 1 - b2_y1)

    ious = inter_area / (b1_area + b2_area - inter_area)
    #iou is 1-D tensor with length n, consists of ious between box1 and n boxes in box2

    return ious

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    #prediction: (batch size, Bbox number, 5+class), first 4 values in prediction: (x, y, w, h)
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    #prediction[:,:,4] > confidence: if the 5th element in dim2 is greater than confidence, the whole dim 2 will become 1
    #conf_mask: (batch size, Bbox number), become (batch size, Bbox number, 1) after unsqueeze
    prediction = prediction * conf_mask #If the confidence element in Bbox is smaller than confidence, set entire row of Bbox to 0
    #Note: * is element-wise product, dot(A,B) is matrix product IMPORTANT!!!!!!!
    #(x, y) in prediction is the coordinate of the center of Bbox, we need to change them into the coordinate of a pair of diagnal corners of each box
    #Change (x, y, w, h) into (x top-left corner, y top-left corner, x bottom-right corner, y bottom-right corner)
    box_corner = prediction.new(prediction.shape) #prediction.new(): define a new tensor with the same data type as prediction
    box_corner[:,:,0] = prediction[:,:,0] - prediction[:,:,2]/2
    box_corner[:,:,1] = prediction[:,:,1] - prediction[:,:,3]/2
    box_corner[:,:,2] = prediction[:,:,0] + prediction[:,:,2]/2
    box_corner[:,:,3] = prediction[:,:,1] + prediction[:,:,3]/2
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write = False

    for i in range(batch_size):
        image_pred = prediction[i] #The prediction of ith image in a batch, image_pred: (Bbox number, class+5)tensor
        max_conf, max_conf_index = torch.max(image_pred[:,5:5+num_classes], 1)
        #torch.max(image_pred[:,5:5+num_classes], 1): find the maximum value along dim1
        max_conf = max_conf.float().unsqueeze(1) #tensor(Bbox number) => tensor(Bbox number, 1)
        max_conf_index = max_conf_index.float().unsqueeze(1)
        image_pred = torch.cat((image_pred[:,:5], max_conf, max_conf_index), 1) #image_pred: tensor(Bbox number, 7)
        non_zero_index = torch.nonzero(image_pred[:,4]) #e.g. the 5th element in 2nd and 4th row are not 0, return tensor([[2],[4]])

        image_pred_new = image_pred[non_zero_index.squeeze(),:].view(-1,7) #view(-1,7): reshape to tensor(num of elements/7, 7)
        #a row of image_pred_new: [x1, y1, x2, y2, score, max confidence, class with max confidence]
        #Note: Score means how likely a Bbox contains an object, confidence is the proberbility that the object is XXX

        if image_pred_new.shape[0] == 0: #If non_zero_index is empty, enter the next loop
            continue

        image_classes = unique(image_pred_new[:,-1]) #Return all the class index appeared image_pred_new, 1-D array
        #unique(): Returns the sorted unique elements of an array

        for c in image_classes:
            equal_c_index = image_pred_new[:,-1] == c
            #e.g. 3 rows in image_pred_new, the last element in row3 = c, equal_c_index = [0, 0, 1]
            class_mask_index = torch.nonzero(equal_c_index).squeeze()
            #Only class c detections left
            image_pred_class = image_pred_new[class_mask_index].view(-1,7)

            #Sort image_pred_class, the entry with maximum score is on the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True)[1]
            #torch.sort(image_pred_class[:,4])[0] is a tensor only contains the sorted 5th column
            #torch.sort(image_pred_class[:,4])[1] contains the indices of the elements in the original input tensor in new order
            image_pred_class = image_pred_class[conf_sort_index]
            num_detections = image_pred_class.size(0) #Number of detections

            #Start to perform nms
            for n in range(num_detections):
                try:
                    ious = bbox_iou(image_pred_class[n].unsqueeze(0), image_pred_class[n + 1:])
                    #image_pred_class[i] is a 1-D tensor with length 7, so it need to be unsqueezed
                except ValueError: #When i is the last detection, there is no image_pred_class[i+1], return ValueError or IndexError
                    break
                except IndexError:
                    break

                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[n+1:] = image_pred_class[n+1:] * iou_mask

                non_zero_index = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_index].view(-1, 7)

            #a row of output: [index of the detected image in the batch, x1, y1, x2, y2, score, max confidence, class with max confidence]
            image_index = image_pred_class.new(image_pred_class.size(0), 1).fill_(i)

            if write == False:
                output = torch.cat((image_index, image_pred_class), 1)
                write = True
            else:
                out = torch.cat((image_index, image_pred_class), 1)
                output = torch.cat((output, out))

    try: #If there is no detection in the batch, there will be no output, return 0
        return output
    except:
        return 0
