# ------------------------------------------------------------------------
# Grounding DINO. Midified by Shilong Liu.
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import contextlib
import copy
import os

import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO

from groundingdino.util.misc import all_gather
from .os_cocoeval import ClsAgnCOCOEval

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]
COCO_INVOC_CATEGORIES = [
    'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
    'motorcycle', 'person', 'potted plant', 'sheep', 'couch',
    'train', 'tv'
]
coco_name_id_dic = {cats["name"]:cats["id"] for cats in COCO_CATEGORIES}
COCO_INVOC_IDS = [coco_name_id_dic[name_cat] for name_cat in COCO_INVOC_CATEGORIES]

GRASPNET_COCO_CATEGORIES = [
    {"color": [220, 20, 60], "id": 1, "name": "cracker_box"},
    {"color": [119, 11, 32], "id": 2, "name": "sugar_box"},
    {"color": [0, 0, 142], "id": 3, "name": "tomato_soup_can"},
    {"color": [0, 0, 230], "id": 4, "name": "mustard_bottle"},
    {"color": [106, 0, 228], "id": 5, "name": "potted_meat_can"},
    {"color": [0, 60, 100], "id": 6, "name": "banana"},
    {"color": [0, 80, 100], "id": 7, "name": "bowl"},
    {"color": [0, 0, 70], "id": 8, "name": "mug"},
    {"color": [0, 0, 192], "id": 9, "name": "power_drill"},
    {"color": [250, 170, 30], "id": 10, "name": "scissors"},
    {"color": [100, 170, 30], "id": 11, "name": "chips_can"},
    {"color": [218, 88, 184], "id": 12, "name": "strawberry"},
    {"color": [220, 220, 0], "id": 13, "name": "apple"},
    {"color": [175, 116, 175], "id": 14, "name": "lemon"},
    {"color": [250, 0, 30], "id": 15, "name": "peach"},
    {"color": [165, 42, 42], "id": 16, "name": "pear"},
    {"color": [255, 77, 255], "id": 17, "name": "orange"},
    {"color": [0, 226, 252], "id": 18, "name": "plum"},
    {"color": [182, 182, 255], "id": 19, "name": "knife"},
    {"color": [0, 82, 0], "id": 20, "name": "phillips_screwdriver"},
    {"color": [120, 166, 157], "id": 21, "name": "flat_screwdriver"},
    {"color": [110, 76, 0], "id": 22, "name": "racquetball"},
    {"color": [174, 57, 255], "id": 23, "name": "b_cups"},
    {"color": [199, 100, 0], "id": 24, "name": "d_cups"},
    {"color": [72, 0, 118], "id": 25, "name": "a_toy_airplane"},
    {"color": [92, 136, 89], "id": 26, "name": "c_toy_airplane"},
    {"color": [255, 179, 240], "id": 27, "name": "d_toy_airplane"},
    {"color": [0, 125, 92], "id": 28, "name": "f_toy_airplane"},
    {"color": [146, 112, 198], "id": 29, "name": "h_toy_airplane"},
    {"color": [210, 170, 100], "id": 30, "name": "i_toy_airplane"},
    {"color": [209, 0, 151], "id": 31, "name": "j_toy_airplane"},
    {"color": [188, 208, 182], "id": 32, "name": "k_toy_airplane"},
    {"color": [0, 220, 176], "id": 33, "name": "padlock"},
    {"color": [255, 99, 164], "id": 34, "name": "dragon"},
    {"color": [92, 0, 73], "id": 35, "name": "secret_repair"},
    {"color": [133, 129, 255], "id": 36, "name": "jvr_cleansing_foam"},
    {"color": [78, 180, 255], "id": 37, "name": "dabao_wash_soup"},
    {"color": [0, 228, 0], "id": 38, "name": "nzskincare_mouth_rinse"},
    {"color": [174, 255, 243], "id": 39, "name": "dabao_sod"},
    {"color": [45, 89, 255], "id": 40, "name": "soap_box"},
    {"color": [134, 134, 103], "id": 41, "name": "kispa_cleanser"},
    {"color": [145, 148, 174], "id": 42, "name": "darlie_toothpaste"},
    {"color": [255, 208, 186], "id": 43, "name": "nivea_men_oil_control"},
    {"color": [197, 226, 255], "id": 44, "name": "baoke_marker"},
    {"color": [168, 171, 172], "id": 45, "name": "hosjam"},
    {"color": [171, 134, 1], "id": 46, "name": "pitcher_cap"},
    {"color": [109, 63, 54], "id": 47, "name": "dish"},
    {"color": [207, 138, 255], "id": 48, "name": "white_mouse"},
    {"color": [151, 0, 95], "id": 49, "name": "camel"},
    {"color": [9, 80, 61], "id": 50, "name": "deer"},
    {"color": [84, 105, 51], "id": 51, "name": "zebra"},
    {"color": [74, 65, 105], "id": 52, "name": "large_elephant"},
    {"color": [166, 196, 102], "id": 53, "name": "rhinocero"},
    {"color": [208, 195, 210], "id": 54, "name": "small_elephant"},
    {"color": [255, 109, 65], "id": 55, "name": "monkey"},
    {"color": [0, 143, 149], "id": 56, "name": "giraffe"},
    {"color": [179, 0, 194], "id": 57, "name": "gorilla"},
    {"color": [209, 99, 106], "id": 58, "name": "weiquan"},
    {"color": [5, 121, 0], "id": 59, "name": "darlie_box"},
    {"color": [227, 255, 205], "id": 60, "name": "soap"},
    {"color": [147, 186, 208], "id": 61, "name": "black_mouse"},
    {"color": [153, 69, 1], "id": 62, "name": "dabao_facewash"},
    {"color": [3, 95, 161], "id": 63, "name": "pantene"},
    {"color": [163, 255, 0], "id": 64, "name": "head_shoulders_supreme"},
    {"color": [119, 0, 170], "id": 65, "name": "thera_med"},
    {"color": [150, 100, 100], "id": 66, "name": "dove"},
    {"color": [0, 182, 199], "id": 67, "name": "head_shoulders_care"},
    {"color": [255, 255, 128], "id": 68, "name": "lion"},
    {"color": [147, 211, 203], "id": 69, "name": "coconut_juice_box"},
    {"color": [0, 165, 120], "id": 70, "name": "hippo"},
    {"color": [191, 162, 208], "id": 71, "name": "tape"},
    {"color": [183, 130, 88], "id": 72, "name": "rubiks_cube"},
    {"color": [95, 32, 0], "id": 73, "name": "peeler_cover"},
    {"color": [130, 114, 135], "id": 74, "name": "peeler"},
    {"color": [110, 129, 133], "id": 75, "name": "ice_cube_mould"},
    {"color": [166, 74, 118], "id": 76, "name": "bar_clamp"},
    {"color": [219, 142, 185], "id": 77, "name": "climbing_hold"},
    {"color": [79, 210, 114], "id": 78, "name": "endstop_holder"},
    {"color": [178, 90, 62], "id": 79, "name": "gearbox"},
    {"color": [65, 70, 15], "id": 80, "name": "mount1"},
    {"color": [127, 167, 115], "id": 81, "name": "mount2"},
    {"color": [59, 105, 106], "id": 82, "name": "nozzle"},
    {"color": [246, 0, 122], "id": 83, "name": "part1"},
    {"color": [142, 108, 45], "id": 84, "name": "part3"},
    {"color": [196, 172, 0], "id": 85, "name": "pawn"},
    {"color": [95, 54, 80], "id": 86, "name": "pipe_connector"},
    {"color": [128, 76, 255], "id": 87, "name": "turbine_housing"},
    {"color": [201, 57, 1], "id": 88, "name": "vase"},
]
GRASPNET_KNOWN_CATEGORIES = [
    "cracker_box", "tomato_soup_can", "banana", "mug", "power_drill", "scissors", "strawberry",
    "peach", "plum", "knife", "flat_screwdriver", "racquetball", "b_cups", "d_toy_airplane",
    "f_toy_airplane", "i_toy_airplane", "j_toy_airplane", "dabao_sod", "darlie_toothpaste",
    "camel", "large_elephant", "rhinocero", "darlie_box", "black_mouse", "dabao_facewash",
    "pantene", "head_shoulders_supreme", "head_shoulders_care"
]
graspnet_known_name_id_dic = {cats["name"]:cats["id"] for cats in GRASPNET_COCO_CATEGORIES}
GRASPNET_KNOWN_IDS = [graspnet_known_name_id_dic[name_cat] for name_cat in GRASPNET_KNOWN_CATEGORIES]


class CocoGroundingEvaluatorClsAgn(object):
    def __init__(self, coco_gt, output_file, iou_types, dataset='coco'):
        assert isinstance(iou_types, (list, tuple))
        if dataset == 'coco':
            known_ids = COCO_INVOC_IDS
        else:
            known_ids = GRASPNET_KNOWN_IDS
        
        coco_gt_k = copy.deepcopy(coco_gt)
        for _, ann in enumerate(coco_gt_k.dataset['annotations']):
            if ann['category_id'] not in known_ids:
                ann['ignore'] = 1
        self.coco_gt_k = coco_gt_k

        coco_gt_unk = copy.deepcopy(coco_gt)
        for _, ann in enumerate(coco_gt_unk.dataset['annotations']):
            if ann['category_id'] in known_ids:
                ann['ignore'] = 1
        self.coco_gt_unk = coco_gt_unk

        self.iou_types = iou_types
        self.coco_eval_k = {}
        self.coco_eval_unk = {}
        for iou_type in iou_types:
            self.coco_eval_k[iou_type] = ClsAgnCOCOEval(coco_gt_k, iouType=iou_type)
            self.coco_eval_k[iou_type].params.useCats = False
            self.coco_eval_k[iou_type].params.maxDets = [10,20,30,50,100,200,300,500,1000]

            self.coco_eval_unk[iou_type] = ClsAgnCOCOEval(coco_gt_unk, iouType=iou_type)
            self.coco_eval_unk[iou_type].params.useCats = False
            self.coco_eval_unk[iou_type].params.maxDets = [10,20,30,50,100,200,300,500,1000]

        self.img_ids = []
        self.eval_imgs_k = {k: [] for k in iou_types}
        self.eval_imgs_unk = {k: [] for k in iou_types}
        self.useCats = False

        self.output_file = output_file

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)

            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt_k = COCO.loadRes(self.coco_gt_k, results) if results else COCO()
                    coco_dt_unk = COCO.loadRes(self.coco_gt_unk, results) if results else COCO()

            coco_eval_k = self.coco_eval_k[iou_type]
            coco_eval_unk = self.coco_eval_unk[iou_type]

            coco_eval_k.cocoDt = coco_dt_k
            coco_eval_k.params.imgIds = list(img_ids)

            coco_eval_unk.cocoDt = coco_dt_unk
            coco_eval_unk.params.imgIds = list(img_ids)

            img_ids_k, eval_imgs_k = evaluate(coco_eval_k)
            img_ids_unk, eval_imgs_unk = evaluate(coco_eval_unk)

            self.eval_imgs_k[iou_type].append(eval_imgs_k)
            self.eval_imgs_unk[iou_type].append(eval_imgs_unk)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs_k[iou_type] = np.concatenate(self.eval_imgs_k[iou_type], 2)
            create_common_coco_eval(self.coco_eval_k[iou_type], self.img_ids, self.eval_imgs_k[iou_type])
            self.eval_imgs_unk[iou_type] = np.concatenate(self.eval_imgs_unk[iou_type], 2)
            create_common_coco_eval(self.coco_eval_unk[iou_type], self.img_ids, self.eval_imgs_unk[iou_type])

    def accumulate(self):
        for coco_eval_k in self.coco_eval_k.values():
            coco_eval_k.accumulate()
        for coco_eval_unk in self.coco_eval_unk.values():
            coco_eval_unk.accumulate()

    def summarize(self):
        for iou_type, coco_eval_k in self.coco_eval_k.items():
            print("KNOWN IoU metric: {}".format(iou_type))
            coco_eval_k.summarize()
            with open(self.output_file, 'a') as f:
                f.write("KNOWN IoU metric: {}\n".format(iou_type))
                f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n".format(coco_eval_k.stats[0]))
                f.write("Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}\n".format(coco_eval_k.stats[1]))
                f.write("Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.3f}\n".format(coco_eval_k.stats[2]))
                f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n".format(coco_eval_k.stats[3]))
                f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n".format(coco_eval_k.stats[4]))
                f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n".format(coco_eval_k.stats[5]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.3f}\n".format(coco_eval_k.stats[6]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = {:.3f}\n".format(coco_eval_k.stats[7]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 30 ] = {:.3f}\n".format(coco_eval_k.stats[8]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 50 ] = {:.3f}\n".format(coco_eval_k.stats[9]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n".format(coco_eval_k.stats[10]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=200 ] = {:.3f}\n".format(coco_eval_k.stats[11]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = {:.3f}\n".format(coco_eval_k.stats[12]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=500 ] = {:.3f}\n".format(coco_eval_k.stats[13]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = {:.3f}\n".format(coco_eval_k.stats[14]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n".format(coco_eval_k.stats[15]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n".format(coco_eval_k.stats[16]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n".format(coco_eval_k.stats[17]))
        for iou_type, coco_eval_unk in self.coco_eval_unk.items():
            print("UNKNOWN IoU metric: {}".format(iou_type))
            coco_eval_unk.summarize()
            with open(self.output_file, 'a') as f:
                f.write("UNKNOWN IoU metric: {}\n".format(iou_type))
                f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n".format(coco_eval_unk.stats[0]))
                f.write("Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}\n".format(coco_eval_unk.stats[1]))
                f.write("Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.3f}\n".format(coco_eval_unk.stats[2]))
                f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n".format(coco_eval_unk.stats[3]))
                f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n".format(coco_eval_unk.stats[4]))
                f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n".format(coco_eval_unk.stats[5]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.3f}\n".format(coco_eval_unk.stats[6]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = {:.3f}\n".format(coco_eval_unk.stats[7]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 30 ] = {:.3f}\n".format(coco_eval_unk.stats[8]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 50 ] = {:.3f}\n".format(coco_eval_unk.stats[9]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n".format(coco_eval_unk.stats[10]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=200 ] = {:.3f}\n".format(coco_eval_unk.stats[11]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = {:.3f}\n".format(coco_eval_unk.stats[12]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=500 ] = {:.3f}\n".format(coco_eval_unk.stats[13]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = {:.3f}\n".format(coco_eval_unk.stats[14]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n".format(coco_eval_unk.stats[15]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n".format(coco_eval_unk.stats[16]))
                f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n".format(coco_eval_unk.stats[17]))


    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] 
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId) 
        for imgId in p.imgIds 
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet) 
        for catId in catIds 
        for areaRng in p.areaRng 
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
