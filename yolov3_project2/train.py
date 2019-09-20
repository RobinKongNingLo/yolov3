from __future__ import division
from darknet import *
from util import *
from terminaltables import AsciiTable
from PIL import ImageFile

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

ImageFile.LOAD_TRUNCATED_IMAGES = True

def arg_parse():

    parser = argparse.ArgumentParser(description = 'YOLO v3 Train Module')
    parser.add_argument("--epochs", dest='epochs', type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", dest='batch_size', type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", dest='gradient_accumulations', type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--cfg", dest='cfgfile', type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_cfg", dest='data_cfg', type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", dest='pretrained_weights', type=str, default = "yolov3.weights", help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", dest='n_cpu', type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", dest='img_size', type=int, default=416, help="size of each image dimension")
    parser.add_argument("--num_classes", dest='num_classes', default=80, help="number of possible classes")
    parser.add_argument("--checkpoint_interval", dest='checkpoint_interval', type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", dest='evaluation_interval', type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", dest='compute_map', default=False, help="if True computes mAP every tenth batch")

    return parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#When exist_ok = False, FileExistsError will be raised if the target directory already exists
os.makedirs('output', exist_ok = True)
os.makedirs('checkpoints', exist_ok = True)

args = arg_parse()
data_cfg = parse_data_config(args.data_cfg)
train_path = data_cfg['train'] #train_path: a list consists of training images
valid_path = data_cfg['valid']
classes = load_classes(data_cfg['names'])
num_classes = data_cfg['classes']

model = Darknet(args.cfgfile).to(device)
model.apply(weights_init_normal)
optimizer = torch.optim.Adam(model.parameters())

if args.pretrained_weights: #If args.pretrained_weights is not empty, return True
    if args.pretrained_weights.endswith('.pth'):
        checkpoint = torch.load(args.pretrained_weights)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print('epoch:', epoch)
        loss = checkpoint['loss']

    else:
        model.load_weights(args.pretrained_weights)

dataset = ListDataset(train_path, augment = True)
#dataset: training images and targets
dataloader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, shuffle = True, pin_memory = True, collate_fn = dataset.collate_fn, )
#pin_memory = True: put the fetched data tensors in pinned memory and thus enables faster data transfer to CUDA-enabled GPUs


metrics = ['grid_size', 'loss', 'x', 'y', 'w', 'h', 'conf', 'cls', 'cls_acc',
           'recall50', 'recall75', 'precision', 'conf_obj', 'conf_noobj', ]

mAP = 0.0

for epoch in range(args.epochs):
    print('Start training, epoch', epoch)
    model.train() #Train mode
    start_time = time.time()
    for i, (_, images, targets) in enumerate(dataloader):
        #i: ith batch
        batches_done = len(dataloader) * epoch + i

        images = Variable(images.to(device))
        targets = Variable(targets.to(device), requires_grad = False)

        loss, outputs = model(images, targets)
        loss.backward()

        if batches_done % args.gradient_accumulations:
            #Accumulate gradient
            optimizer.step()
            optimizer.zero_grad()

        #Log process
        log_str = '\n---- [Epoch %d/%d, Batch %d/%d] ----\n' % (epoch, args.epochs, i, len(dataloader))
        log_str += f'\nTotal loss {loss.item()}'
        log_str += f'\nLast mAP: {mAP}'

        #Approximate time left for epoch
        epoch_batches_left = len(dataloader) - (i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (i + 1))
        log_str += f'\n---- ETA {time_left}'

        print(log_str)

        #model.seen += images.size(0) #model.seen: training pictures number

    if epoch % args.evaluation_interval == 0:
        print('\n---- Evaluating Model ----')
        #Evaluate model on validation set
        precision, recall, AP, f1, ap_class = evaluate(
            model, path = valid_path, iou_thres = 0.5, conf_thres = 0.5,
            nms_thres = 0.5, image_size = args.img_size, batch_size = 8, num_classes = args.num_classes)

        ap_table = [['Index', 'Class name', 'AP']]
        for i, c in enumerate(ap_class):
            ap_table += [[c, classes[c], '%.5f' % AP[i]]]
        mAP = AP.mean()
        print(AsciiTable(ap_table).table)
        print(f'---- mAP {mAP}')

    if epoch % args.checkpoint_interval == 0:
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                    f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
