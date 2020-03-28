import os
import re
import sys
sys.path.append('.')
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from config import update_config, cfg,pafprocess
from enum import Enum
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]  # = 19
CocoPairsRender = CocoPairs[:-2]

def draw_humans(npimg, humans, imgcopy=False):
    if imgcopy:
        npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    lines = []
    for human in humans:
        # draw point
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue

            # npimg = cv2.line(npimg, centers[pair[0]], centers[pair[1]], common.CocoColors[pair_order], 3)
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)
            lines.append(centers[pair[0]][0])
            lines.append(centers[pair[0]][1])
            lines.append(centers[pair[1]][0])
            lines.append(centers[pair[1]][1])
            #print(centers[pair[1]])

    lines = np.array(lines)
    lines = ' '.join(str(i) for i in lines)
    #print(lines)
    return npimg, centers, lines

def compute_resized_coords(coords, resizeFactor):
    return (np.array(coords, dtype=float) + 0.5) * resizeFactor - 0.5

def find_peaks(param, img):
    peaks_binary = (maximum_filter(img, footprint=generate_binary_structure(
        2, 1)) == img) * (img > param)
    # Note reverse ([::-1]): we return [[x y], [x y]...] instead of [[y x], [y
    # x]...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T

def NMS(heatmaps, upsampFactor=1., bool_refine_center=True, bool_gaussian_filt=False, config=None):
    joint_list_per_joint_type = []
    cnt_total_joints = 0

    win_size = 2

    for joint in range(config.MODEL.NUM_KEYPOINTS):
        map_orig = heatmaps[:, :, joint]
        peak_coords = find_peaks(config.TEST.THRESH_HEATMAP, map_orig)
        peaks = np.zeros((len(peak_coords), 4))
        for i, peak in enumerate(peak_coords):
            if bool_refine_center:
                x_min, y_min = np.maximum(0, peak - win_size)
                x_max, y_max = np.minimum(
                    np.array(map_orig.T.shape) - 1, peak + win_size)

                # Take a small patch around each peak and only upsample that
                # tiny region
                patch = map_orig[y_min:y_max + 1, x_min:x_max + 1]
                map_upsamp = cv2.resize(
                    patch, None, fx=upsampFactor, fy=upsampFactor, interpolation=cv2.INTER_CUBIC)

                # Gaussian filtering takes an average of 0.8ms/peak (and there might be
                # more than one peak per joint!) -> For now, skip it (it's
                # accurate enough)
                map_upsamp = gaussian_filter(
                    map_upsamp, sigma=3) if bool_gaussian_filt else map_upsamp

                # Obtain the coordinates of the maximum value in the patch
                location_of_max = np.unravel_index(
                    map_upsamp.argmax(), map_upsamp.shape)
                # Remember that peaks indicates [x,y] -> need to reverse it for
                # [y,x]
                location_of_patch_center = compute_resized_coords(
                    peak[::-1] - [y_min, x_min], upsampFactor)
                # Calculate the offset wrt to the patch center where the actual
                # maximum is
                refined_center = (location_of_max - location_of_patch_center)
                peak_score = map_upsamp[location_of_max]
            else:
                refined_center = [0, 0]
                # Flip peak coordinates since they are [x,y] instead of [y,x]
                peak_score = map_orig[tuple(peak[::-1])]
            peaks[i, :] = tuple(
                x for x in compute_resized_coords(peak_coords[i], upsampFactor) + refined_center[::-1]) + (
                              peak_score, cnt_total_joints)
            cnt_total_joints += 1
        joint_list_per_joint_type.append(peaks)

    return joint_list_per_joint_type


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()

class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])



def paf_to_pose_cpp(heatmaps, pafs, config):
    humans = []
    joint_list_per_joint_type = NMS(heatmaps, upsampFactor=config.MODEL.DOWNSAMPLE, config=config)

    joint_list = np.array(
        [tuple(peak) + (joint_type,) for joint_type, joint_peaks in enumerate(joint_list_per_joint_type) for peak in
         joint_peaks]).astype(np.float32)

    if joint_list.shape[0] > 0:
        joint_list = np.expand_dims(joint_list, 0)
        paf_upsamp = cv2.resize(
            pafs, None, fx=config.MODEL.DOWNSAMPLE, fy=config.MODEL.DOWNSAMPLE, interpolation=cv2.INTER_NEAREST)
        heatmap_upsamp = cv2.resize(
            heatmaps, None, fx=config.MODEL.DOWNSAMPLE, fy=config.MODEL.DOWNSAMPLE, interpolation=cv2.INTER_NEAREST)
        pafprocess.process_paf(joint_list, heatmap_upsamp, paf_upsamp)
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False
            for part_idx in range(config.MODEL.NUM_KEYPOINTS):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue
                is_added = True
                human.body_parts[part_idx] = BodyPart(
                    '%d-%d' % (human_id, part_idx), part_idx,
                    float(pafprocess.get_part_x(c_idx)) / heatmap_upsamp.shape[1],
                    float(pafprocess.get_part_y(c_idx)) / heatmap_upsamp.shape[0],
                    pafprocess.get_part_score(c_idx)
                )
            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

    return humans


def rtpose_preprocess(image):
    image = image.astype(np.float32)
    image = image / 256. - 0.5
    image = image.transpose((2, 0, 1)).astype(np.float32)

    return image


def vgg_preprocess(image):
    image = image.astype(np.float32) / 255.
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = image.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]

    preprocessed_img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
    return preprocessed_img


def inception_preprocess(image):
    image = image.copy()[:, :, ::-1]
    image = image.astype(np.float32)
    image = image / 128. - 1.
    image = image.transpose((2, 0, 1)).astype(np.float32)

    return image



def ssd_preprocess(image):
    image = image.astype(np.float32)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image -= (104.0, 117.0, 123.0)

    processed_img = rgb_image.astype(np.float32)
    processed_img = processed_img[:, :, ::-1].copy()
    processed_img = processed_img.transpose((2, 0, 1)).astype(np.float32)

    return processed_img

def _factor_closest(num, factor, is_ceil=True):
    num = np.ceil(float(num) / factor) if is_ceil else np.floor(float(num) / factor)
    num = int(num) * factor
    return num


def crop_with_factor(im, dest_size=None, factor=32, is_ceil=True):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.
    # if max_size is not None and im_size_min > max_size:
    im_scale = float(dest_size) / im_size_min
    im = cv2.resize(im, None, fx=im_scale, fy=im_scale)

    h, w, c = im.shape
    new_h = _factor_closest(h, factor=factor, is_ceil=is_ceil)
    new_w = _factor_closest(w, factor=factor, is_ceil=is_ceil)
    im_croped = np.zeros([new_h, new_w, c], dtype=im.dtype)
    im_croped[0:h, 0:w, :] = im

    return im_croped, im_scale, im.shape



def get_outputs(img, model, preprocess):
    """Computes the averaged heatmap and paf for the given image
    :param multiplier:
    :param origImg: numpy array, the image being processed
    :param model: pytorch model
    :returns: numpy arrays, the averaged paf and heatmap
    """
    inp_size = cfg.DATASET.IMAGE_SIZE

    # padding
    im_croped, im_scale, real_shape = crop_with_factor(
        img, inp_size, factor=cfg.MODEL.DOWNSAMPLE, is_ceil=True)

    if preprocess == 'rtpose':
        im_data = rtpose_preprocess(im_croped)

    elif preprocess == 'vgg':
        im_data = vgg_preprocess(im_croped)

    elif preprocess == 'inception':
        im_data = inception_preprocess(im_croped)

    elif preprocess == 'ssd':
        im_data = ssd_preprocess(im_croped)

    batch_images= np.expand_dims(im_data, 0)

    # several scales as a batch
    batch_var = torch.from_numpy(batch_images).cuda().float()
    predicted_outputs, _ = model(batch_var)
    output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
    heatmap = output2.cpu().data.numpy().transpose(0, 2, 3, 1)[0]
    paf = output1.cpu().data.numpy().transpose(0, 2, 3, 1)[0]

    return paf, heatmap, im_scale


def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)


def make_vgg19_block(block):
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)



def get_model(trunk='vgg19'):
    """Creates the whole CPM model
    Args:
        trunk: string, 'vgg19' or 'mobilenet'
    Returns: Module, the defined model
    """
    blocks = {}
    # block0 is the preprocessing stage
    if trunk == 'vgg19':
        block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
                  {'conv1_2': [64, 64, 3, 1, 1]},
                  {'pool1_stage1': [2, 2, 0]},
                  {'conv2_1': [64, 128, 3, 1, 1]},
                  {'conv2_2': [128, 128, 3, 1, 1]},
                  {'pool2_stage1': [2, 2, 0]},
                  {'conv3_1': [128, 256, 3, 1, 1]},
                  {'conv3_2': [256, 256, 3, 1, 1]},
                  {'conv3_3': [256, 256, 3, 1, 1]},
                  {'conv3_4': [256, 256, 3, 1, 1]},
                  {'pool3_stage1': [2, 2, 0]},
                  {'conv4_1': [256, 512, 3, 1, 1]},
                  {'conv4_2': [512, 512, 3, 1, 1]},
                  {'conv4_3_CPM': [512, 256, 3, 1, 1]},
                  {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    elif trunk == 'mobilenet':
        block0 = [{'conv_bn': [3, 32, 2]},  # out: 3, 32, 184, 184
                  {'conv_dw1': [32, 64, 1]},  # out: 32, 64, 184, 184
                  {'conv_dw2': [64, 128, 2]},  # out: 64, 128, 92, 92
                  {'conv_dw3': [128, 128, 1]},  # out: 128, 256, 92, 92
                  {'conv_dw4': [128, 256, 2]},  # out: 256, 256, 46, 46
                  {'conv4_3_CPM': [256, 256, 1, 3, 1]},
                  {'conv4_4_CPM': [256, 128, 1, 3, 1]}]

    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - 6
    for i in range(2, 7):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        ]

    models = {}

    if trunk == 'vgg19':
        print("Bulding VGG19")
        models['block0'] = make_vgg19_block(block0)

    for k, v in blocks.items():
        models[k] = make_stages(list(v))

    class rtpose_model(nn.Module):
        def __init__(self, model_dict):
            super(rtpose_model, self).__init__()
            self.model0 = model_dict['block0']
            self.model1_1 = model_dict['block1_1']
            self.model2_1 = model_dict['block2_1']
            self.model3_1 = model_dict['block3_1']
            self.model4_1 = model_dict['block4_1']
            self.model5_1 = model_dict['block5_1']
            self.model6_1 = model_dict['block6_1']

            self.model1_2 = model_dict['block1_2']
            self.model2_2 = model_dict['block2_2']
            self.model3_2 = model_dict['block3_2']
            self.model4_2 = model_dict['block4_2']
            self.model5_2 = model_dict['block5_2']
            self.model6_2 = model_dict['block6_2']

            self._initialize_weights_norm()

        def forward(self, x):

            saved_for_loss = []
            out1 = self.model0(x)

            out1_1 = self.model1_1(out1)
            out1_2 = self.model1_2(out1)
            out2 = torch.cat([out1_1, out1_2, out1], 1)
            saved_for_loss.append(out1_1)
            saved_for_loss.append(out1_2)

            out2_1 = self.model2_1(out2)
            out2_2 = self.model2_2(out2)
            out3 = torch.cat([out2_1, out2_2, out1], 1)
            saved_for_loss.append(out2_1)
            saved_for_loss.append(out2_2)

            out3_1 = self.model3_1(out3)
            out3_2 = self.model3_2(out3)
            out4 = torch.cat([out3_1, out3_2, out1], 1)
            saved_for_loss.append(out3_1)
            saved_for_loss.append(out3_2)

            out4_1 = self.model4_1(out4)
            out4_2 = self.model4_2(out4)
            out5 = torch.cat([out4_1, out4_2, out1], 1)
            saved_for_loss.append(out4_1)
            saved_for_loss.append(out4_2)

            out5_1 = self.model5_1(out5)
            out5_2 = self.model5_2(out5)
            out6 = torch.cat([out5_1, out5_2, out1], 1)
            saved_for_loss.append(out5_1)
            saved_for_loss.append(out5_2)

            out6_1 = self.model6_1(out6)
            out6_2 = self.model6_2(out6)
            saved_for_loss.append(out6_1)
            saved_for_loss.append(out6_2)

            return (out6_1, out6_2), saved_for_loss

        def _initialize_weights_norm(self):

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.normal_(m.weight, std=0.01)
                    if m.bias is not None:  # mobilenet conv2d doesn't add bias
                        init.constant_(m.bias, 0.0)

            # last layer of these block don't have Relu
            init.normal_(self.model1_1[8].weight, std=0.01)
            init.normal_(self.model1_2[8].weight, std=0.01)

            init.normal_(self.model2_1[12].weight, std=0.01)
            init.normal_(self.model3_1[12].weight, std=0.01)
            init.normal_(self.model4_1[12].weight, std=0.01)
            init.normal_(self.model5_1[12].weight, std=0.01)
            init.normal_(self.model6_1[12].weight, std=0.01)

            init.normal_(self.model2_2[12].weight, std=0.01)
            init.normal_(self.model3_2[12].weight, std=0.01)
            init.normal_(self.model4_2[12].weight, std=0.01)
            init.normal_(self.model5_2[12].weight, std=0.01)
            init.normal_(self.model6_2[12].weight, std=0.01)

    model = rtpose_model(models)
    return model




parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='/root/Person_Pose_Estimation/experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='/root/Person_Pose_Estimation/demo/pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)

model = get_model('vgg19')
model.load_state_dict(torch.load(args.weight))
model.cuda()
model.float()
model.eval()

order = ''

def run_image(oriImg):
    shape_dst = np.min(oriImg.shape[0:2])

    with torch.no_grad():
        paf, heatmap, imscale = get_outputs(
            oriImg, model, 'rtpose')

    humans = paf_to_pose_cpp(heatmap, paf, cfg)

    out, center, order = draw_humans(oriImg, humans)

    #print(order)

    point = []
    for i in center:
        point.append(center[i][0])
        point.append(center[i][1])

    point = np.array(point)
    point = ' '.join(str(i) for i in point)
    point = point + ' '
    order = order + ' '

    # Display the resulting frame
    cv2.imshow('Video', out)
    cv2.waitKey(1)

    return point, order


def run():
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        oriImg = video_capture.read()[1]

        shape_dst = np.min(oriImg.shape[0:2])

        with torch.no_grad():
            paf, heatmap, imscale = get_outputs(
                oriImg, model, 'rtpose')

        humans = paf_to_pose_cpp(heatmap, paf, cfg)

        out, center, order = draw_humans(oriImg, humans)

        print(order)

        point = []
        # Display the resulting frame
        cv2.imshow('Video', out)


        for i in center:
            point.append(center[i][0])
            point.append(center[i][1])
            #print(center[i])
        point = np.array(point)
        point = ' '.join(str(i) for i in point)

        test_string = 'nihaoma'

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    return point, test_string
