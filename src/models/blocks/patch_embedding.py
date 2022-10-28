# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Path Embedding module."""

from mindspore import nn
import mindspore.ops as ops


class PatchEmbedding(nn.Cell):
    """
    Path embedding layer for cswin. First rearrange b c (h p) (w p) -> b (h w) (p p c).

    Args:
        image_size (int): Input image size. Default: 224.
        embed_dim (int): Patch size of image. Default: 64.
        k_size (int): k_size for 2D convolution.
        stride (int): stride for 2D convolution
        input_channels (int): The number of input channel. Default: 3.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = PathEmbedding(224, 64, 7, 4, 3)
    """

    def __init__(self,
                 image_size: int = 224,
                 embed_dim: int = 64,
                 k_size: int = 7,
                 stride: int = 4, 
                 in_chans: int = 3,
                 padding: int = 2,
                 norm_layer=nn.LayerNorm):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size

        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size= k_size, stride= stride, pad_mode = "pad", padding = padding, has_bias=True)
        self.norm = norm_layer((embed_dim,))


    def construct(self, x):
        """
        Path Embedding construct.
        """
        x = self.conv(x)
        b, c, h, w = x.shape
        x = ops.Reshape()(x, (b, c, h * w))
        x = ops.Transpose()(x, (0, 2, 1))
        x = self.norm(x)
        return x
