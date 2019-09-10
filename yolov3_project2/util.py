from __future__ import division

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

import tqdm
import os

def to_cpu(tensor):
    return tensor.detach().cpu()

def parse_data_config(path):

    options = dict()
    options['gpus'] = '0, 1, 2, 3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'): #Ignore empty line and comments
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

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

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1 = dim_diff // 2
    pad2 = dim_diff - dim_diff // 2
    if h <= w:
        pad = (0, 0, pad1, pad2)
    else:
        pad = (pad1, pad2, 0, 0)
    img = F.pad(img, pad, 'constant', value = pad_value)

    return img, pad

def horizontal_flip(images, targets):
    images = torch.flip(images, [-1]) #Flip images horizontally
    targets[:, 2] = 1 - targets[:, 2] #x -> 1 - x
    return images, targets

class ListDataset(Dataset):
    def __init__(self, list_path, image_size = 416, augment = True, normalized_labels = True):
        with open(list_path, 'r') as file:
            self.image_files = file.readlines()
            #img_files: a list consists of image directories

        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.image_files]
        self.image_size = image_size
        self.max_objects = 100
        self.augment = augment
        self.normalized_labels = normalized_labels
        self.min_size = self.image_size - 3 * 32
        self.max_size = self.image_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):
        #targets[i] is a tensor consists of bounding boxes of ith image
        image_path = self.image_files[index % len(self.image_files)].rstrip()
        #Extract image as tensor
        image = transforms.ToTensor()(Image.open(image_path).convert('RGB')) #.convert('RGB'): can convert greyscale image into RGB image
        _, h, w = image.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        #Pad images to square
        image, pad = pad_to_square(image, 0)
        _, padded_h, padded_w = image.shape

        label_path = self.label_files[index % len(self.image_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            #boxes is a (num_boxes, 5) tensor, each line represents a bounding box
            #A row of boxes: [class, x, y, w, h], x, y, w, h are with respect to the original image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            #Coordinate with respect to padded image
            x1 = x1 + pad[0]
            y1 = y1 + pad[2]
            x2 = x2 + pad[1]
            y2 = y2 + pad[3]
            #(x, y, w, h) respect to padded image
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes #The 1st element of each row is 0, a row of targets: [0, class, x, y, w, h]

        if self.augment: #Enable augmentation will increase the diversity of training data
            if np.random.random() < 0.5:
                image, targets = horizontal_flip(image, targets) #Filp img and targets horizontally to generate extra data

        return image_path, image, targets

    def collate_fn(self, batch):
        #Use own collate_fn to process the list of samples to form a batch, batch is a list with all the examples
        paths, images, targets = list(zip(*batch)) #Unzip paths, images, targets from __getitem__
        #Remove empty targets
        targets = [boxes for boxes in targets if boxes is not None]
        #Add image index to the 1st element of each row of targets
        for i, boxes in enumerate(targets):
            #boxes: a tensor consists of all the bounding boxes in ith image
            boxes[..., 0] = i
        #Change targets from a tuple of tensors into a single tensor, each row represents a bounding box
        targets = torch.cat(targets, 0)
        #Resize images into input shape. F.interpolate: resize image
        images = torch.stack([F.interpolate(image.unsqueeze(0), size=self.image_size, mode="nearest").squeeze(0) for image in images])
        self.batch_count += 1

        return paths, images, targets

    def __len__(self):
        return len(self.image_files)

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    #pred_boxes: [batches, anchors, grid_size, grid_size, box_parameters]
    #target: [num_of_boxes, 6], a row of target: [0, class, x, y, w, h]
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    n_batch = pred_boxes.size(0)
    n_anchor = pred_boxes.size(1)
    n_class = pred_cls.size(-1)
    n_grid = pred_boxes.size(2)

    obj_mask = ByteTensor(n_batch, n_anchor, n_grid, n_grid).fill_(0)
    noobj_mask = ByteTensor(n_batch, n_anchor, n_grid, n_grid).fill_(1)
    class_mask = FloatTensor(n_batch, n_anchor, n_grid, n_grid).fill_(0)
    iou_scores = FloatTensor(n_batch, n_anchor, n_grid, n_grid).fill_(0)
    tx = FloatTensor(n_batch, n_anchor, n_grid, n_grid).fill_(0)
    ty = FloatTensor(n_batch, n_anchor, n_grid, n_grid).fill_(0)
    tw = FloatTensor(n_batch, n_anchor, n_grid, n_grid).fill_(0)
    th = FloatTensor(n_batch, n_anchor, n_grid, n_grid).fill_(0)
    tcls = FloatTensor(n_batch, n_anchor, n_grid, n_grid, n_class).fill_(0)

    target_boxes = target[:, 2:6] * n_grid
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    #Get anchors with best iou with target boxes
    ious = torch.stack([anchor_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    #best_n: a tensor consists of indecies of anchors with best iou with target boxes
    #Separate target values
    b, target_labels = target[:, :2].long().t() #.t(): seprate the columns of tensor .long(): keep integer part
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    #gi, gj are the coordinate of grids
    gi, gj = gxy.long().t()
    #Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0
    #Set noobj mask to 0 where iou > ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    #one-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    #Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    #pred_boxes and targets are both x, y, w, h. x1y1x2y2 = False
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2 = False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

def anchor_iou(anchor, wh):
    wh = wh.t()
    w1, h1 = anchor[0], anchor[1]
    w2, h2 = wh[0], wh[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = w1 * h1 + w2 * h2 - inter_area
    iou = inter_area / union_area
    return iou

def bbox_iou(box1, box2, x1y1x2y2 = True):

    if not x1y1x2y2:
        #Transform from xywh to x1y1x2y2
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        #box1, box2 are (1,7) and (n,7) tensors
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
        #b1_x1... are (1,1) tensors
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
        #b2_x1... are (n,1) tensors

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    #inter_x1... are (n,1) tensors

    inter_area = torch.clamp(inter_x2 + 1 - inter_x1, min = 0) * torch.clamp(inter_y2 +1 - inter_y1, min = 0)
    #Because we are dealing with pixels, the bottom-right coordinate of intersection is (inter_x2+1, inter_y2+1)
    #torch.clamp(x, min=0): if x < 0, return 0

    b1_area = (b1_x2 + 1 - b1_x1) * (b1_y2 + 1 - b1_y1)
    b2_area = (b2_x2 + 1 - b2_x1) * (b2_y2 + 1 - b2_y1)

    ious = inter_area / (b1_area + b2_area - inter_area)
    #iou is 1-D tensor with length n, consists of ious between box1 and n boxes in box2

    return ious

def evaluate(model, path, iou_thres, conf_thres, nms_thres, image_size, batch_size, num_classes):
    #path: the path of validation set
    model.eval()

    dataset = ListDataset(path, image_size = image_size, augment = False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False, collate_fn = dataset.collate_fn)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = [] #List of (true_pos, pred_scores, pred_labels)
    for i, (_, images, targets) in enumerate(tqdm.tqdm(dataloader, desc='Detecting objects')): #tqdm: progress bar
        #i: ith batch
        #Extract classes to labels
        labels = labels + targets[:, 1].tolist()
        #(x y w h) to (x1 y1 x2 y2)
        targets_corner = targets.new(targets.shape)
        targets_corner[..., 2] = targets[..., 2] - targets[..., 4] / 2
        targets_corner[..., 3] = targets[..., 3] - targets[..., 5] / 2
        targets_corner[..., 4] = targets[..., 2] + targets[..., 4] / 2
        targets_corner[..., 5] = targets[..., 3] + targets[..., 5] / 2
        #Rescale target
        targets[:, 2:] = targets_corner[:, 2:]
        targets[:, 2:] *= image_size

        images = Variable(images.type(Tensor), requires_grad = False)
        targets = Variable(targets.type(Tensor))

        with torch.no_grad(): #torch.no_grad(): impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations.
            outputs = model(images)

        outputs = write_results(outputs, confidence = conf_thres, num_classes = num_classes, nms_conf = 0.4)

        sample_metrics += get_batch_statistics(outputs, targets, iou_thres = iou_thres)

    #true_positives, false_negative, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    true_pred, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_pred, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

def get_batch_statistics(outputs, targets, iou_thres):
    #Compute true positive, predicted scores and predicted labels per sample
    #Each row of outputs represent a bounding box: [index of the detected image in the batch, x1, y1, x2, y2, score, max confidence, class with max confidence]
    batch_metrics = []

    for sample_i in range(int(outputs[-1, 0]) + 1): #Outputs[-1, 0] is the index of the last image in the batch

        output = outputs[outputs[:, 0] == sample_i][:, 1:]
        #A row of output: [x1, y1, x2, y2, score, max confidence, class with max confidence]
        if len(output) == 0:
            continue
        pred_boxes = output[..., :4]
        pred_scores = output[..., 4]
        pred_labels = output[..., 6]

        true_pred = torch.zeros(pred_boxes.shape[0])
        #Each row of target represent a target bounding box: [label, x, y, w, h]
        target = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = target[:, 0] if len(target) else []

        if len(target):
            detected_boxes = []
            target_boxes = target[:, 1:]

            for box_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                if len(detected_boxes) == len(target):
                    break

                #Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                #box_index: the index of target box with the highest iou with the predicted box
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)

                if iou >= iou_thres and box_index not in detected_boxes and pred_label == target_labels[box_index]:
                    true_pred[box_i] = 1

        batch_metrics.append([true_pred, pred_scores, pred_labels])

    return batch_metrics

def ap_per_class(true_pred, pred_scores, pred_class, target_class):
    #Compute the average precision of each class
    #Sort the metrics from highest pred_scores to the lowest pred_scores
    i = torch.argsort(pred_scores, descending = True)
    true_pred, pred_scores, pred_class = true_pred[i], pred_scores[i], pred_class[i]

    #List all the classes appeared in target_class
    unique_classes = np.unique(target_class)

    #Create Precision-Recall curve and compute AP for each class
    ap, precision, recall = [], [], []
    for c in tqdm.tqdm(unique_classes, desc = 'Computing AP'):
        i = pred_class == c #i is the index of predicted bounding boxes with class c
        num_targ = (target_class == c).sum() #Number of target boxes with class c
        num_pred = i.sum() #Number of predicted boxes with class c


        if num_pred == 0 and num_targ == 0:
            continue

        elif num_pred == 0 or num_targ == 0:
            ap.append(0)
            recall.append(0)
            precision.append(0)

        else:
            '''
            Accumulate FP and TP
            cumsum(): accumulating
            e.g. cumsum([1, 2, 4, 4]) = [1, 3, 7, 11]
            '''

            false_pred_curv = torch.cumsum((1 - true_pred[i]), dim = 0)
            true_pred_curv = torch.cumsum(true_pred[i], dim = 0)

            #Recall: ratio of true object detections to the total number of objects in the dataset
            recall_curv = true_pred_curv / (num_targ + 1e-16)
            recall.append(recall_curv[-1])

            #Precision: ratio of true object detections to the total number of objects detections
            precision_curv = true_pred_curv / (true_pred_curv + false_pred_curv)
            precision.append(precision_curv[-1])

            ap.append(compute_ap(recall_curv, precision_curv))

    #Compute F1 score
    precision, recall, ap = np.array(precision), np.array(recall), np.array(ap)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)

    return precision, recall, ap, f1, unique_classes.astype('int32')

def compute_ap(recall_curv, precision_curv):
    #Compute average precision from precision-recall curve (ap is the area under precision-recall curve)
    #Append sentinel values (sentinel value is a special value in the context of an algorithm which uses its presence as a condition of termination)
    mrec = np.concatenate(([0.0], recall_curv, [1.0]))
    mpre = np.concatenate(([0.0], precision_curv, [0.0]))

    #Interpolated precision
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    #Look for points where X-axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

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
