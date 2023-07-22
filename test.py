"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import copy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import pandas as pd
from PIL import Image
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.2, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default= 0.2, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=2400, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=50, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()


""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]


    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def merge_bboxes(bboxes, delta_x=0.1, delta_y=0.18):
    """
    Arguments:
        bboxes {list} -- list of bounding boxes with each bounding box is a list [xmin, ymin, xmax, ymax]
        delta_x {float} -- margin taken in width to merge
        detlta_y {float} -- margin taken in height to merge
    Returns:
        {list} -- list of bounding boxes merged
    """

    def is_in_bbox(point, bbox):
        """
        Arguments:
            point {list} -- list of float values (x,y)
            bbox {list} -- bounding box of float_values [xmin, ymin, xmax, ymax]
        Returns:
            {boolean} -- true if the point is inside the bbox
        """
        return point[0] >= bbox[0] and point[0] <= bbox[2] and point[1] >= bbox[1] and point[1] <= bbox[3]

    def intersect(bbox, bbox_):
        """
        Arguments:
            bbox {list} -- bounding box of float_values [xmin, ymin, xmax, ymax]
            bbox_ {list} -- bounding box of float_values [xmin, ymin, xmax, ymax]
        Returns:
            {boolean} -- true if the bboxes intersect
        """
        for i in range(int(len(bbox) / 2)):
            for j in range(int(len(bbox) / 2)):
                # Check if one of the corner of bbox inside bbox_
                if is_in_bbox([bbox[2 * i], bbox[2 * j + 1]], bbox_):
                    return True
        return False
    
    def good_intersect(bbox, bbox_):
        return (bbox[0] < bbox_[2] and bbox[2] > bbox_[0] and
                bbox[1] < bbox_[3] and bbox[3] > bbox_[1])

    # Sort bboxes by ymin
    bboxes = sorted(bboxes, key=lambda x: x[1])

    tmp_bbox = None
    while True:
        nb_merge = 0
        used = []
        new_bboxes = []
        # Loop over bboxes
        for i, b in enumerate(bboxes):
            for j, b_ in enumerate(bboxes):
                # If the bbox has already been used just continue
                if i in used or j <= i:
                    continue
                # Compute the bboxes with a margin
                bmargin = [
                    b[0] - (b[2] - b[0]) * delta_x, b[1] - (b[3] - b[1]) * delta_y,
                    b[2] + (b[2] - b[0]) * delta_x, b[3] + (b[3] - b[1]) * delta_y
                ]
                b_margin = [
                    b_[0] - (b_[2] - b_[0]) * delta_x, b_[1] - (b[3] - b[1]) * delta_y,
                    b_[2] + (b_[2] - b_[0]) * delta_x, b_[3] + (b_[3] - b_[1]) * delta_y
                ]
                # Merge bboxes if bboxes with margin have an intersection
                # Check if one of the corner is in the other bbox
                # We must verify the other side away in case one bounding box is inside the other
                if intersect(bmargin, b_margin) or intersect(b_margin, bmargin):
                    tmp_bbox = [min(b[0], b_[0]), min(b[1], b_[1]), max(b_[2], b[2]), max(b[3], b_[3])]
                    used.append(j)
                    # print(bmargin, b_margin, 'done')
                    nb_merge += 1
                if tmp_bbox:
                    b = tmp_bbox
            if tmp_bbox:
                new_bboxes.append(tmp_bbox)
            elif i not in used:
                new_bboxes.append(b)
            used.append(i)
            tmp_bbox = None
        # If no merge were done, that means all bboxes were already merged
        if nb_merge == 0:
            break
        bboxes = copy.deepcopy(new_bboxes)

    return new_bboxes

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    if ang == 0:
        ang = -90
    else:
        ang = -ang
    return ang #+ 360 if ang < 0 else ang

if __name__ == '__main__':
    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image_ocr = cv2.imread(image_path) 
        image = imgproc.loadImage(image_path)
        # gray =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)

        # save text detection results 
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)
        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)
        res_file = result_folder + "res_" + filename + '.txt'
####################################Post detection process and recognition#################################################
        all_boxes = bboxes
        bboxes = []
        # get xmin ymin xmax ymax as coordinates  
        for llis in bboxes:
            x = [llis[0], llis[1], llis[4], llis[5]]
            bboxes.append(x)
        # merge overlapping and nearby bounding boxes
        bboxes = merge_bboxes(bboxes, delta_x=0.2, delta_y=0.08)
        count = 1
        all_text = ""
        for box in bboxes:
            # draw merged bounding boxes on image
            cv2.rectangle(image_ocr, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
            # scline bbox and convert to text with pytesseract
            arr = image[box[1]:box[3], box[0]:box[2]] 
            try:
                im = Image.fromarray(arr)
                text = pytesseract.image_to_string(im)
                all_text = all_text + "\n" + text
            except:
                count = count + 1
                pass
        # save image to folder
        image_bb = Image.fromarray(image_ocr)
        image_bb.save("result/inter_bbox_image_{}.txt".format(filename))
        # filter angled bounding boxes
        df = pd.DataFrame(all_boxes, columns = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4" ])
        cross_bboxes = np.array(df[(df.x1 != df.x4) | (abs(df.y1 - df.y3) > 350)])
        for box in cross_bboxes:
            pt1 = (int((box[2] + box[4])/2), int((box[3] + box[5])/2))
            pt2 = (int((box[0] + box[6])/2), int((box[1] + box[7])/2))
            pt3 = (pt2[0] + 700, pt2[1]) 
            # calculate angle of the bbounding box with x axis from left to right
            angle = getAngle(pt1, pt2, pt3)
            if (angle < -20) or (angle) > 20:
                pts = np.array(box).reshape(-1, 2)
                mask = np.zeros(image.shape[0:2], dtype=np.uint8)
                points = pts
                # method 1 smooth region
                cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
                res = cv2.bitwise_and(image,image,mask = mask)
                rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
                cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
                im = Image.fromarray(cropped)
                # rotate image
                out = im.rotate(angle, expand=True)
                text = pytesseract.image_to_string(out)
                all_text = all_text + text
        
        text_file = open("result/out_text_{}.txt".format(filename), "w")
        #write string to file
        text_file.write(all_text)
        
        #close file
        text_file.close()
#################################################################################################################
    print("elapsed time : {}s".format(time.time() - t))