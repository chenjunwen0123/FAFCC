"""
WS-DAN models

Hu et al.,
"See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classification",
arXiv:1901.09891

Created: May 04,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
'''
修改记录：fpn,权重
'''
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import models.vgg as vgg
import models.resnet as resnet
from models.inception import inception_v3, BasicConv2d

__all__ = ['WSDAN']
EPSILON = 1e-12

#sttention map可视化
ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
visualize = True
if visualize:
    os.makedirs('./FGVC/Stanford-Cars/atten_visualize/', exist_ok=True)
    
def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)

# class MyLayer(torch.nn.Module):
#     def __init__(self,M):

#         super(MyLayer,self).__init__()
#         self.M = M
#         self.par_list = [torch.nn.Parameter(torch.randn(())) for i in range(self.M)]
#         print('par_list is ',self.par_list)
#     def forward(self, feature_matrix_list):
#         final_list = [self.par_list[i]*feature_matrix_list[i] for i in range(self.M)]
#         return final_list

# Bilinear Attention Pooling
class BAP(nn.Module):#GAP全局平均池化，GMP全局最大池化
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)
#         self.mylayer = MyLayer(32)

    def forward(self, features, attentions):
        B, C, H, W = features.size()#batch_size（一次训练所提取的样本数量） 通道数 高 宽
        _, M, AH, AW = attentions.size()

        # match size如果特征图和注意力图大小不一致，使其变一致
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)特征矩阵变化
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)#爱因斯坦求和
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
#             feature_matrix = self.mylayer(feature_matrix)
            feature_matrix = torch.cat(feature_matrix, dim=1)#两个矩阵拼接

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix


# WS-DAN: Weakly Supervised Data Augmentation Network for FGVC
class WSDAN(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e', pretrained=False):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

        # Network Initialization
        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'vgg' in net:
            self.features = getattr(vgg, net)(pretrained=pretrained).get_features()
            self.num_features = 512
        elif 'resnet' in net:
            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion
        else:
            raise ValueError('Unsupported net: %s' % net)



        # Attention Maps  1*1卷积进行特征降维
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        
        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        
        # 添加另一路 featureA
        self.features_v1 = inception_v3(pretrained=pretrained).get_features_mixed_7c()
        self.num_features_v1 = 2048
        self.attentions_v1 = BasicConv2d(self.num_features_v1, self.M, kernel_size=1)
        #减少7c通道数
        self.top_layer = BasicConv2d(2048,768,kernel_size=1)
        #消除混叠效应，3*3卷积
        self.smooth = BasicConv2d(768,768,kernel_size=3, padding=1)
        
        

        # Classification Layer
        #self.fc = nn.Linear(self.M * self.num_features+self.M * self.num_features_v1, self.num_classes, bias=False)
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

        logging.info('WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes, self.M))

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        
        ## 另一路7c层
        feature_maps_v1 = self.features_v1(x)
        #上采样代码
        #转换尺寸2048-》768
        p7c = self.top_layer(feature_maps_v1)
        p6e = self.upsample_add(p7c,feature_maps)
        p6e = self.smooth(p6e)
        feature_maps = p6e
        
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]
        feature_matrix = self.bap(feature_maps, attention_maps)

        '''
        # 另一路
        #feature_maps_v1 = self.features_v1(x)
        attention_maps_v1 = feature_maps_v1[:, :self.M, ...]
        feature_matrix_v1 = self.bap(feature_maps_v1, attention_maps_v1)
        '''
        '''
        #attention可视化
        if visualize:
            # reshape attention maps
            attention_map = F.upsample_bilinear(attention_maps, size=(x.size(2), x.size(3)))
            attention_map = torch.sqrt(attention_maps.cpu() / attention_map.max().item())
            attention_maps1 = F.upsample_bilinear(attention_maps_v1, size=(x.size(2), x.size(3)))
            attention_maps1 = torch.sqrt(attention_maps1.cpu() / attention_maps1.max().item())

            # get heat attention maps
            heat_attention_maps = generate_heatmap(attention_map)
            heat_attention_maps1 = generate_heatmap(attention_maps1)

            # raw_image, heat_attention, raw_attention
            raw_image = x.cpu() * STD + MEAN
            heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
            raw_attention_image = raw_image * attention_map
            heat_attention_image1 = raw_image * 0.5 + heat_attention_maps1 * 0.5
            raw_attention_image1 = raw_image * attention_maps1

            for batch_idx in range(X.size(0)):
                    #rimg = ToPILImage(raw_image[batch_idx])
                    raimg = ToPILImage(raw_attention_image[batch_idx])
                    haimg = ToPILImage(heat_attention_image[batch_idx])
                    #rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * config.batch_size + batch_idx)))
                    raimg.save(os.path.join(savepath, '%03d_6e_atten.jpg' % (i * config.batch_size + batch_idx)))
                    haimg.save(os.path.join(savepath, '%03d_6e_atten.jpg' % (i * config.batch_size + batch_idx)))
                    #rimg = ToPILImage(raw_image[batch_idx])
                    raimg1 = ToPILImage(raw_attention_image1[batch_idx])
                    haimg1 = ToPILImage(heat_attention_image1[batch_idx])
                    #rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * config.batch_size + batch_idx)))
                    raimg1.save(os.path.join(savepath, '%03d_7c_atten.jpg' % (i * config.batch_size + batch_idx)))
                    haimg1.save(os.path.join(savepath, '%03d_7c_atten.jpg' % (i * config.batch_size + batch_idx)))
        '''
        # 合并两个特征
        #final_feature_matrix = torch.cat([feature_matrix,feature_matrix_v1],1)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        if self.training:
            # Randomly choose one of attention maps Ak
            attention_map = []
            for i in range(batch_size):
                attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
                attention_weights = F.normalize(attention_weights, p=1, dim=0)
                k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
                attention_map.append(attention_maps[i, k_index, ...])
            attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        else:
            # Object Localization Am = mean(Ak)
            attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        # p: (B, self.num_classes)
        # feature_matrix: (B, M * C)
        # attention_map: (B, 2, H, W) in training, (B, 1, H, W) in val/testing
        return p, feature_matrix, attention_map

    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN, self).load_state_dict(model_dict)
        
    #上采样，相加
    def upsample_add(self,x,y):
        _,_,H,W = y.shape
        return F.upsample(x,size=(H,W),mode='bilinear') + y