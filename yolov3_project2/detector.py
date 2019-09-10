from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random

from terminaltables import AsciiTable

def arg_parse():

    '''
    Using argparse means the doesn’t need to go into the code and make changes
    to the script. Users can edit the variables in cmd. Giving the user the
    ability to enter command line arguments provides flexibility.
    '''

    parser = argparse.ArgumentParser(description = 'YOLO v3 Detection Module')

    parser.add_argument('--images', dest = 'images', help =
                        'Image / Directory containing images to perform detection upon',
                        default = 'images/val20142', type = str)
    parser.add_argument('--det', dest = 'det', help =
                        'Image / Directory to store detections to',
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help =
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help =
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)

    return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes('data/coco.names')

print('Loading network.....')
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print('Network successfully loaded')

model.net_info['height'] = args.reso
input_dim = int(model.net_info['height'])
#If input_dim % 32 is not 0 or input_dim not greater than 32, give AssertionError
assert input_dim % 32 == 0
assert input_dim > 32

if CUDA:
    model.cuda()

model.eval() #Set to evaluation mode, dropout and BN are different during evaluation and training

data_cfg = parse_data_config('config/coco.data')
valid_path = data_cfg['valid']
precision, recall, AP, f1, ap_class = evaluate(
    model, path = valid_path, iou_thres = 0.5, conf_thres = 0.5,
    nms_thres = 0.5, image_size = 416, batch_size = 8, num_classes = 80)

ap_table = [['Index', 'Class name', 'AP']]
for i, c in enumerate(ap_class):
    ap_table += [[c, classes[c], '%.5f' % AP[i]]]
print(AsciiTable(ap_table).table)
print(f'---- mAP {AP.mean()}')

read_dir = time.time()

try:
    #osp.join(osp.realpath('.'), 'x', 'y') returns current_working_file\x\y, osp.realpath('.') returns the current working file
    #imlist is a list of all images in images file
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError: #If the path above is wrong, return current_working_file\images
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()

if not os.path.exists(args.det): #If the directory defined by det does not exits, create it
    os.makedirs(args.det)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

#loaded_ims is the list consists of all the images
#e.g. inp_dim = 3, len(imlist) = 5, [inp_dim for x in range(len(imlist))] = [3, 3, 3, 3, 3]
#map((prep_image, loaded_ims, [inp_dim for x in range(len(imlist))])) performs prep_image() on every elements in loaded_ims
im_batches = list(map(prep_image, loaded_ims, [input_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims] #A list of sizes of the original Images
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2) #Transform im_dim_list from list [(W, H)] into FloatTensor [[W, H, W, H]], used to scale ((x1, y1, x2, y2)) later

#Divide im_dim_list into batches. batch_size: how many images in one batch
leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover
    im_batches = [torch.cat((im_batches[i * batch_size : min((i + 1) * batch_size, len(im_batches))])) for i in range(num_batches)]
    #New im_batches is a list, an element of the list is a list consists of images in one batch

write = 0

if CUDA:
    im_dim_list = im_dim_list.cuda()

start_det_loop = time.time()
for i, batch in enumerate(im_batches):
    #i: in ith batch
    start = time.time()
    if CUDA:
        batch = batch.cuda()

    with torch.no_grad(): #When not using backward(), no_grad() will reduce memory consumption
        #prediction: (batch_size, 10647, 85)
        prediction = model(Variable(batch))

    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)
    #Return the bounding boxes after NMS, bounding box: (ind, x1, y1, x2, y2, s, s_cls, index_cls)
    end = time.time()

    #Prediction is 0 means there is no object detected in ith batch.
    if type(prediction) == int:
        for n, image in enumerate(imlist[i * batch_size : min((i + 1) * batch_size, len(im_batches))]):
            image_id = i * batch_size + n
            print('{0:20s} predicted in {1:6.3f} seconds'.format(image.split('/')[-1], (end - start) / batch_size))
            print('{0:20s} {1:s}'.format('Objects Detected:', ''))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i * batch_size #Change the index of images in a batch into the index of images in all images
    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat((output, prediction)) #Add new predictions below old predictions

    for n, image in enumerate(imlist[i * batch_size : min((i + 1) * batch_size, len(im_batches))]):
        image_id = i * batch_size + n
        objects = [classes[int(x[-1])] for x in output if int(x[0]) == image_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objects)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize

#Drawing bounding boxes on images
try:
    output
except NameError:
    print('No object detected')
    exit()

output_recast = time.time()

#Need to transform the corner attributes of each bounding box to original dimensions of images
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
#torch.index_select(input, dim, index): Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
#The dimth dimension has the same size as the length of index; other dimensions have the same size as in the original tensor.
scaling_factor = torch.min(input_dim / im_dim_list, 1)[0].view(-1, 1)
#torch.min(inpit_dim / im_dim_list, 1)[0]: a tensor consists of the minimum value in each row (dim1)
#scaling_factor: (num of output, 1) tensor
output[:, [1, 3]] = output[:, [1, 3]] - (input_dim - scaling_factor * im_dim_list[:, 0].view(-1,1))/2
#x1 = x1 - (416 - scaling_factor * image_w)/2, x2 = x2 - (416 - scaling_factor * image_w)/2
output[:, [2, 4]] = output[:, [2, 4]] - (input_dim - scaling_factor * im_dim_list[:, 1].view(-1,1))/2
#y1 = y1 - (416 - scaling_factor * image_h)/2, y2 = y2 - (416 - scaling_factor * image_h)/2
#(x1, y1) and (x2, y2) are the coordinate with respect to top-left corner of resized image (without padding)
output[:,1:5] = output[:,1:5] / scaling_factor
#Resize (x1, y1) and (x2, y2) with respect to original image

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])
    #Clip any bounding boxes that may have boundaries outside the image to the edges of our image.

class_load = time.time()
colors = pkl.load(open('pallete', 'rb'))
#pallete is a pickled file that contains many colors to randomly choose from

#draw the boxes
draw = time.time()

def write(x, images):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = images[int(x[0])]
    #Load the image related to the bounding box
    cls = int(x[-1])
    color = random.choice(colors)
    label = '{0}'.format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    #cv2.getTextSize(text, fontFace, fontScale, thickness): (textSize, baseline)
    #textSize: the size of a box that contains the specified text.
    c3 = (c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4)
    cv2.rectangle(img, c1, c3, color, -1) #-1: create a filled rectangle
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] +4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

print('output:', output)
list(map(lambda x: write(x, loaded_ims), output))
#lambda x: a row of output (a bounding box), run write(x, loaded_ims) on each bounding box, draw bounding boxes on original images
det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("\\")[-1]))
#Create a list of addresses, to which we will save detection images to
list(map(cv2.imwrite, det_names, loaded_ims))
#Write the images with detections to the address in det_names
end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
