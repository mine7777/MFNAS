'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid

import PlainNet
from PlainNet import _get_right_parentheses_index_
from PlainNet.super_blocks import PlainNetSuperBlockClass
from torch import nn
import global_utils

class SuperResConv(PlainNetSuperBlockClass):

    expansion = 4

    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=1, 
                 no_create=False, no_reslink=False, no_BN=False, use_se=False, **kwargs):
        super(SuperResConv, self).__init__(**kwargs)
        self.planes = out_channels // self.expansion
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sub_layers = sub_layers
        self.no_create = no_create
        self.use_se = use_se
        if self.use_se:
            print('---debug use_se in ' + str(self))

        full_str = ''

        for i in range(self.sub_layers):
            inner_str = ''

            #downsample
            # if self.stride > 1:
            #     full_str += 'AvgPool({},{})'.format(2, self.stride)
            
            # projection 
            inner_str += 'ConvKX({},{},{},{})'.format(self.in_channels, self.planes, 1, 1)
            inner_str += 'ConvDW({},{},{})'.format(self.planes, 3, self.stride)
            inner_str += 'BN({})'.format(self.planes)

            #attention net
            #FIXME:reserve or remove
            inner_str += 'RELU({})'.format(self.planes)

            inner_str += 'ConvKX({},{},{},{})'.format(self.planes, self.planes, 3, 1)
            inner_str += 'BN({})'.format(self.planes)
            inner_str += 'RELU({})'.format(self.planes)

            #res link
            inner_str += 'ConvKX({},{},{},{})'.format(self.planes, self.out_channels, 1, 1)
            inner_str += 'BN({})'.format(self.out_channels)
            inner_str += 'RELU({})'.format(self.out_channels)
            if self.in_channels != self.out_channels:
                res_str = 'ResBlockProj({},{},{})RELU({})'.format(self.in_channels, self.stride, inner_str, self.out_channels)
            else:
                res_str = 'ResBlock({},{})RELU({})'.format(self.in_channels,inner_str, self.out_channels)
            full_str += res_str
        pass

        self.block_list = PlainNet.create_netblock_list_from_str(full_str, no_create=no_create, no_reslink=no_reslink, no_BN=no_BN, **kwargs)
        if not no_create:
            self.module_list = nn.ModuleList(self.block_list)
        else:
            self.module_list = None

    def __str__(self):
        return type(self).__name__ + '({},{},{})'.format(self.in_channels, self.out_channels, self.stride)

    def __repr__(self):
        return type(self).__name__ + '({}|in={},out={},stride={})'.format(
            self.block_name, self.in_channels, self.out_channels, self.stride
        )

    def split(self, split_layer_threshold):
        return str(self)

    def structure_scale(self, scale=1.0, channel_scale=None, sub_layer_scale=None):
        if channel_scale is None:
            channel_scale = scale
        if sub_layer_scale is None:
            sub_layer_scale = scale

        new_out_channels = global_utils.smart_round(self.out_channels * channel_scale)

        return type(self).__name__ + '({},{},{})'.format(self.in_channels, new_out_channels,
                                                               self.stride)


    @classmethod
    def create_from_str(cls, s, **kwargs):
        assert cls.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len(cls.__name__ + '('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        return cls(in_channels=in_channels, out_channels=out_channels, stride=stride,
                   block_name=tmp_block_name, **kwargs),s[idx + 1:]
    

class SuperResAtten(PlainNetSuperBlockClass):
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=1 ,sub_layers=1, kernel_size=None,
                 no_create=False, no_reslink=False, no_BN=False, use_se=False, heads = 4, dim_head=64, **kwargs):
        super(SuperResAtten, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bottleneck_channels = bottleneck_channels
        self.sub_layers = sub_layers
        self.kernel_size = kernel_size
        #TODO: to be modified
        self.heads = heads
        self.no_create = no_create
        self.no_reslink = False
        self.no_BN = no_BN
        self.use_se = use_se
        if self.use_se:
            print('---debug use_se in ' + str(self))

        full_str = ''

        # TODO: to be modified
        attn_dim_in = out_channels // 4

        attn_dim_out = dim_head * heads

        for i in range(self.sub_layers):
            inner_str = ''

            # if self.stride > 1:
            #     full_str += 'ConvKX({},{},{},{})'.format(self.in_channels, self.out_channels, 3, self.stride)
            #     full_str += 'BN({})'.format(self.out_channels)
            #     full_str += 'RELU({})'.format(self.out_channels)

            # projection 
            inner_str += 'ConvKX({},{},{},{})'.format(self.in_channels, attn_dim_in, 1, 1)
            inner_str += 'ConvDW({},{},{})'.format(attn_dim_in, 3, self.stride)
            inner_str += 'BN({})'.format(attn_dim_in)

            #attention net
            inner_str += 'RELU({})'.format(attn_dim_in)
            inner_str += 'MSA({},{},{})'.format(attn_dim_in, heads, dim_head)
            inner_str += 'BN({})'.format(attn_dim_out)
            inner_str += 'RELU({})'.format(attn_dim_out)
            inner_str += 'ConvKX({},{},{},{})'.format(attn_dim_out, out_channels, 1, 1)
            inner_str += 'BN({})'.format(out_channels)

            #res link
            if self.in_channels != self.out_channels:
                res_str = 'ResBlockProj({},{},{})RELU({})'.format(self.in_channels, self.stride, inner_str, out_channels)
            else:
                res_str = 'ResBlock({},{})RELU({})'.format(self.in_channels,inner_str, self.out_channels)
            full_str += res_str
        pass

        self.block_list = PlainNet.create_netblock_list_from_str(full_str, no_create=no_create, no_reslink=no_reslink, no_BN=no_BN, **kwargs)
        if not no_create:
            self.module_list = nn.ModuleList(self.block_list)
        else:
            self.module_list = None

    def __str__(self):
        return type(self).__name__ + '({},{},{})'.format(self.in_channels, self.out_channels, self.stride)

    def __repr__(self):
        return type(self).__name__ + '({}|in={},out={},stride={})'.format(
            self.block_name, self.in_channels, self.out_channels, self.stride
        )

    def split(self, split_layer_threshold):
        if self.sub_layers >= split_layer_threshold:
            new_sublayers_1 = split_layer_threshold // 2
            new_sublayers_2 = self.sub_layers - new_sublayers_1
            new_block_str1 = type(self).__name__ + '({},{},{})'.format(self.in_channels, self.out_channels,
                                                                self.stride)
            new_block_str2 = type(self).__name__ + '({},{},{})'.format(self.out_channels, self.out_channels,
                                                                self.stride)
            return new_block_str1 + new_block_str2
        else:
            return str(self)

    def structure_scale(self, scale=1.0, channel_scale=None, sub_layer_scale=None):
        if channel_scale is None:
            channel_scale = scale
        if sub_layer_scale is None:
            sub_layer_scale = scale

        new_out_channels = global_utils.smart_round(self.out_channels * channel_scale)

        return type(self).__name__ + '({},{},{})'.format(self.in_channels, new_out_channels,
                                                               self.stride)


    @classmethod
    def create_from_str(cls, s, **kwargs):
        assert cls.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len(cls.__name__ + '('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        return cls(in_channels=in_channels, out_channels=out_channels, stride=stride,
                   block_name=tmp_block_name, **kwargs),s[idx + 1:]


def register_netblocks_dict(netblocks_dict: dict):
    this_py_file_netblocks_dict = {
        'SuperResAtten': SuperResAtten,
        'SuperResConv': SuperResConv,
    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    return netblocks_dict