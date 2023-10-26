from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention as LlamaAttentionHF

from punica.models.llama import LlamaAttention as LlamaAttentionPunica
from punica.models.llama_lora import LlamaAttentionWithLora as LlamaAttentionPunicaLora, \
    LlamaMlpWithLora as LlamaMlpPunicaLora


class ReplaceWithTensorSlicing:
    def __init__(self, mp_group=None, mp_size=1, out_dim=1, in_dim=0):
        if mp_group is not None:
            self.gpu_index = dist.get_rank(group=mp_group)
        else:
            self.gpu_index = 0
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mp_size = mp_size

    def merge_assert(self, dim1, dim2):
        assert dim1 > dim2, \
            'Merging tensors is not allowed here! Please use deepspeed load_checkpoint\
            for merging your checkpoints before replacing the transformer layer with\
            inference-kernels'

    def strided_copy(self,
                     dst: Optional[torch.Tensor],
                     src: Optional[torch.Tensor],
                     num_splits: int,
                     int8: bool = False,
                     allocate_tensor: bool = False):
        if src is None:
            return src
        src_shape = src.shape
        dst_shape = dst.shape

        outer_dim = 0 if int8 else -1

        if allocate_tensor:
            dst = torch.empty_like(dst)

        src_split = torch.split(src.data, src.shape[outer_dim] // num_splits, dim=outer_dim)
        if (len(src_shape) == 2 and len(dst_shape) == 2):
            if src_shape[outer_dim] == dst_shape[self.out_dim]:
                dst = dst.reshape(-1).data.copy_(src.data.reshape(-1)).reshape(src.shape)
                dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
                if hasattr(src, 'scale'):
                    dst.scale = src.scale
                return dst
            self.merge_assert(src_shape[outer_dim], dst_shape[self.out_dim])
            qkv_size = dst_shape[self.out_dim] // num_splits
            qkv_split = [torch.split(src_s, qkv_size, dim=outer_dim) for src_s in src_split]
            weight_split = [
                torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=outer_dim) for i in range(len(qkv_split[0]))
            ]
            dst = dst.reshape(-1).data.copy_(weight_split[self.gpu_index].contiguous().reshape(-1)).reshape(
                weight_split[self.gpu_index].shape)
        else:
            if src_shape[0] == dst_shape[0]:
                return torch.nn.parameter.Parameter(src)
            qkv_size = dst_shape[0] // num_splits
            qkv_split = [torch.split(src_s, qkv_size, dim=0) for src_s in src_split]
            bias_split = [torch.cat([qkv_s[i] for qkv_s in qkv_split], axis=0) for i in range(len(qkv_split[0]))]
            dst.data.copy_(bias_split[self.gpu_index].contiguous())

        dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
        if hasattr(src, 'scale'):
            dst.scale = src.scale
        return dst

    def copy(self, dst, src, int8=False, allocate_tensor=False):
        if src is None:
            return src
        assert not dst.data.is_meta  # the torch.Tensor.copy_ method used below will silently fail on meta tensors
        if allocate_tensor:
            dst = torch.empty_like(dst)
        outer_dim = 0 if int8 else 1
        inner_dim = 1 if int8 else 0
        src_shape = src.shape
        dst_shape = dst.shape
        if (len(src_shape) == 2 and len(dst_shape) == 2):

            if src_shape[inner_dim] == dst_shape[self.in_dim] and src_shape[outer_dim] == dst_shape[self.out_dim]:
                dst = dst.reshape(-1).data.copy_(src.data.reshape(-1)).reshape(src.shape)
            else:
                if src_shape[inner_dim] != dst_shape[self.in_dim]:
                    self.merge_assert(src_shape[inner_dim], dst_shape[self.in_dim])
                    dst.data.copy_(src[:, self.gpu_index * dst_shape[self.in_dim]: (self.gpu_index + 1) * dst_shape[self.in_dim]] if inner_dim == 1 else \
                                   src[self.gpu_index * dst_shape[self.in_dim]: (self.gpu_index + 1) * dst_shape[self.in_dim], :])
                else:
                    self.merge_assert(src_shape[outer_dim], dst_shape[self.out_dim])
                    dst.data.copy_(src[:, self.gpu_index * dst_shape[self.out_dim]: (self.gpu_index + 1) * dst_shape[self.out_dim]] if outer_dim == 1 else \
                                   src[self.gpu_index * dst_shape[self.out_dim]: (self.gpu_index + 1) * dst_shape[self.out_dim], :])
        else:
            if src_shape[0] == dst_shape[0]:
                dst = src if src.dtype == dst.dtype else dst.data.copy_(src)
            else:
                dst.data.copy_(src[self.gpu_index * dst_shape[-1]:(self.gpu_index + 1) * dst_shape[-1]])
        dst = torch.nn.parameter.Parameter(dst, requires_grad=False)
        if hasattr(src, 'scale'):
            dst.scale = src.scale
        return dst


class LinearAllReduce(nn.Module):
    def __init__(self, weight, bias=None, mp_group=None):
        super(LinearAllReduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.all_reduce(output, group=self.mp_group)
        if self.bias is not None:
            output += self.bias
        return output


class LinearLayer(nn.Module):
    def __init__(self, weight=None, bias=None):
        super(LinearLayer, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias
        return output


class AutoTP:
    def __init__(self, mp_group, mp_size):
        self.mp_group = mp_group
        self.mp_size = mp_size

    def update_mp_params(self, child):
        if getattr(child, "replaced", False) == True:
            return
        if isinstance(child, LlamaMLP) or isinstance(child, LlamaMlpPunicaLora):
            params_list = ["intermediate_size"]
        elif isinstance(child, LlamaAttentionHF):
            params_list = ["hidden_size", "num_heads", "num_key_value_heads"]
        elif isinstance(child, LlamaAttentionPunica) or isinstance(child, LlamaAttentionPunicaLora):
            params_list = ["hidden_size", "num_qo_heads", "num_kv_heads"]
        for param in params_list:
            if hasattr(child, param):
                param_val = getattr(child, param)
                assert param_val % self.mp_size == 0, f"{param} ({param_val}) must be divisible by mp_size ({self.mp_size})"
                setattr(child, param, param_val // self.mp_size)
        setattr(child, "replaced", True)

    def replace_linear_all_reduce(self, child):
        if getattr(child, "replaced", False) == True:
            return
        weight_shape = child.weight.shape
        mp_replace = ReplaceWithTensorSlicing(mp_group=self.mp_group, mp_size=self.mp_size)
        data = child.weight.data.split(weight_shape[1] // self.mp_size, dim=1)
        data = data[mp_replace.gpu_index]
        setattr(child, "replaced", True)
        return LinearAllReduce(nn.Parameter(data, requires_grad=False),
                               bias=None if child.bias is None else nn.Parameter(child.bias),
                               mp_group=self.mp_group)

    def replace_linear(self, child):
        if getattr(child, "replaced", False) == True:
            return
        weight_shape = child.weight.shape
        mp_replace = ReplaceWithTensorSlicing(mp_group=self.mp_group, mp_size=self.mp_size)
        data = child.weight.data.split(weight_shape[0] // self.mp_size, dim=0)
        data = data[mp_replace.gpu_index]
        if child.bias is not None:
            bias_data = child.bias.data.split(weight_shape[0] // self.mp_size, dim=0)
            bias_data = bias_data[mp_replace.gpu_index]
            bias_data = nn.Parameter(bias_data, requires_grad=False)
        else:
            bias_data = None
        setattr(child, "replaced", True)
        return LinearLayer(weight=nn.Parameter(data, requires_grad=False), bias=bias_data)

    def replace_module(self, r_module):
        for _, child in r_module.named_children():
            if isinstance(child, LlamaMLP) or isinstance(child, LlamaMlpPunicaLora):
                setattr(child, "down_proj", self.replace_linear_all_reduce(child.down_proj))
                setattr(child, "gate_proj", self.replace_linear(child.gate_proj))
                setattr(child, "up_proj", self.replace_linear(child.up_proj))
            elif isinstance(child, LlamaAttentionHF) or isinstance(child, LlamaAttentionPunica) \
                or isinstance(child, LlamaAttentionPunicaLora):
                setattr(child, "q_proj", self.replace_linear(child.q_proj))
                setattr(child, 'k_proj', self.replace_linear(child.k_proj))
                setattr(child, 'v_proj', self.replace_linear(child.v_proj))
                setattr(child, 'o_proj', self.replace_linear_all_reduce(child.o_proj))
            else:
                self.replace_module(child)
                continue
            self.update_mp_params(child)
        return r_module
