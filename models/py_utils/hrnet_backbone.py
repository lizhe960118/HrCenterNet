# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import pdb
import torch
import os
import functools

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr

from .kp_utils import _tranpose_and_gather_feat, _decode
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss
from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer, make_ct_layer
# from .kp_utils import make_pool_layer, make_unpool_layer
# from .kp_utils import make_merge_layer, make_inter_layer
from .kp_utils import make_cnv_layer

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], 
                                       momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, 
                                            momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

class hr(nn.Module):
    def __init__(
        self, 
        db,  
        out_dim,
        n_stack = 1,
        pre=None, 
        cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer, make_ct_layer=make_ct_layer,
        make_cnv_layer=make_cnv_layer, 
        make_heat_layer=make_kp_layer, make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer

    ):
        # 对于hourglass定义的
        # n = 5;  
        # nstack 104 -> 2 52->1; 这个等于2, 可以消去
        # dims    = [256, 256, 384, 384, 384, 512]
        # modules = [2, 2, 2, 2, 2, 4]
        # out_dim = 80;
        # make_up_layer, make_low_layer, make_hg_layer, make_hg_layer_revr, 
        # make_pool_layer, make_unpool_layer, make_merge_layer, make_inter_layer, kp_layer

        # make_tl_layer None
        # make_br_layer None
        # make_ct_layer None
        # make_cnv_layer convolution(3, inp_dim, out_dim)
        # make_kp_layer convolution(3, cnv_dim, curr_dim, with_bn=False), nn.Conv2d(curr_dim, out_dim, (1, 1))

        super(hr, self).__init__()

        self._decode            = _decode
        self._db                = db
        self.K                  = self._db.configs["top_k"]
        self.ae_threshold       = self._db.configs["ae_threshold"]
        self.kernel             = self._db.configs["nms_kernel"]
        self.input_size         = self._db.configs["input_size"][0]
        self.output_size        = self._db.configs["output_sizes"][0][0]

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ) if pre is None else pre

        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # 这里需要替换为 hrnet
        self.stage2_cfg = {'NUM_MODULES':1, 'NUM_BRANCHES':2, 'BLOCK': 'BASIC', 'NUM_BLOCKS':[4, 4], 'NUM_CHANNELS': [18, 36], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = {'NUM_MODULES':4, 'NUM_BRANCHES':3, 'BLOCK': 'BASIC', 'NUM_BLOCKS':[4,4,4], 'NUM_CHANNELS': [18, 36, 72], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = {'NUM_MODULES':3, 'NUM_BRANCHES':4, 'BLOCK': 'BASIC', 'NUM_BLOCKS':[4, 4, 4, 4], 'NUM_CHANNELS': [18, 36, 72, 144], 'FUSE_METHOD': 'SUM'}
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # 这里返回时，curr_dim要有所确定
        # curr_dim = ? # 这里是经过hrnet之后的
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        # 之后确立各种分支
        curr_dim = last_inp_channels
        self.cnvs = make_cnv_layer(curr_dim, cnv_dim)  
     

        self.tl_cnvs = make_tl_layer(cnv_dim)  
       
        self.br_cnvs = make_br_layer(cnv_dim)  
     

        self.ct_cnvs =  make_ct_layer(cnv_dim) 

        ## keypoint heatmaps
        self.tl_heats = make_heat_layer(cnv_dim, curr_dim, out_dim) 

        self.br_heats = make_heat_layer(cnv_dim, curr_dim, out_dim)  

        self.ct_heats = make_heat_layer(cnv_dim, curr_dim, out_dim)  


        ## tags
        self.tl_tags  = make_tag_layer(cnv_dim, curr_dim, 1)  

        self.br_tags  = make_tag_layer(cnv_dim, curr_dim, 1)
        
        self.tl_heats[-1].bias.data.fill_(-2.19)
        self.br_heats[-1].bias.data.fill_(-2.19)
        self.ct_heats[-1].bias.data.fill_(-2.19)


        self.tl_regrs = make_regr_layer(cnv_dim, curr_dim, 2)  

        self.br_regrs = make_regr_layer(cnv_dim, curr_dim, 2)  

        self.ct_regrs = make_regr_layer(cnv_dim, curr_dim, 2)

        self.relu = nn.ReLU(inplace=True)
       
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _train(self, *xs):
        image      = xs[0]
        tl_inds    = xs[1]
        br_inds    = xs[2]
        ct_inds    = xs[3]

        inter      = self.pre(image)
        inter      = self.layer1(inter)
        x = inter
        outs       = []
    
        # hr_ = self.hrs
        cnv_ = self.cnvs
        tl_cnv_ = self.tl_cnvs
        br_cnv_  = self.br_cnvs
        ct_cnv_ = self.ct_cnvs
        tl_heat_ = self.tl_heats
        br_heat_ = self.br_heats
        ct_heat_ = self.ct_heats
        tl_tag_ = self.tl_tags
        br_tag_ = self.br_tags
        tl_regr_ = self.tl_regrs
        br_regr_ = self.br_regrs
        ct_regr_ = self.ct_regrs

        # hr = hr_(inter)
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)
        cnv = cnv_(x) # 这里需要进一步实现， 出来的就是具有高分辨率的特征图

        tl_cnv = tl_cnv_(cnv)
        br_cnv = br_cnv_(cnv)
        ct_cnv = ct_cnv_(cnv)

        tl_heat, br_heat, ct_heat = tl_heat_(tl_cnv), br_heat_(br_cnv), ct_heat_(ct_cnv)
        tl_tag, br_tag        = tl_tag_(tl_cnv),  br_tag_(br_cnv)
        tl_regr, br_regr, ct_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), ct_regr_(ct_cnv)

        tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
        br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)

        outs += [tl_heat, br_heat, ct_heat, tl_tag, br_tag, tl_regr, br_regr, ct_regr]

        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        inter = self.layer1(inter)
        x = inter

        outs          = []

        cnv_ = self.cnvs
        tl_cnv_ = self.tl_cnvs
        br_cnv_  = self.br_cnvs
        ct_cnv_ = self.ct_cnvs
        tl_heat_ = self.tl_heats
        br_heat_ = self.br_heats
        ct_heat_ = self.ct_heats
        tl_tag_ = self.tl_tags
        br_tag_ = self.br_tags
        tl_regr_ = self.tl_regrs
        br_regr_ = self.br_regrs
        ct_regr_ = self.ct_regrs

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2, x3], 1)

        cnv = cnv_(x)

        tl_cnv = tl_cnv_(cnv)
        br_cnv = br_cnv_(cnv)
        ct_cnv = ct_cnv_(cnv)

        tl_heat, br_heat, ct_heat = tl_heat_(tl_cnv), br_heat_(br_cnv), ct_heat_(ct_cnv)
        tl_tag, br_tag        = tl_tag_(tl_cnv),  br_tag_(br_cnv)
        tl_regr, br_regr, ct_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), ct_regr_(ct_cnv)

        outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr,
                    ct_heat, ct_regr]
                
        return self._decode(*outs[-8:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        stride = 8

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        ct_heats = outs[2::stride]
        tl_tags  = outs[3::stride]
        br_tags  = outs[4::stride]
        tl_regrs = outs[5::stride]
        br_regrs = outs[6::stride]
        ct_regrs = outs[7::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_ct_heat = targets[2]
        gt_mask    = targets[3]
        gt_tl_regr = targets[4]
        gt_br_regr = targets[5]
        gt_ct_regr = targets[6]
        
        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]
        ct_heats = [_sigmoid(c) for c in ct_heats]
        
#         print(tl_heats[0].size()) # torch.Size([6, 80, 128, 128])
#         print(br_heats[0].size()) # torch.Size([6, 256, 128, 128])
#         print(ct_heats[0].size()) # torch.Size([6, 256, 128, 128])
        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)
        focal_loss += self.focal_loss(ct_heats, gt_ct_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr, ct_regr in zip(tl_regrs, br_regrs, ct_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
            regr_loss += self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(tl_heats)
        return loss.unsqueeze(0), (focal_loss / len(tl_heats)).unsqueeze(0), (pull_loss / len(tl_heats)).unsqueeze(0), (push_loss / len(tl_heats)).unsqueeze(0), (regr_loss / len(tl_heats)).unsqueeze(0)
