
import math

import mindspore
from mindspore import nn, Tensor
from mindspore.common import initializer

from mindspore import ops
from mindspore.ops import operations as P

import numpy as np

from src.models.blocks import Mlp, CSWinBlock, PatchEmbedding, Merge_Block
from src.models.blocks.weight_init import KaimingNormal, TruncNormal
from src.models.blocks.misc import Identity

def _cfg(url='', **kwargs):
    return {
        'num_classes': 1000, 
        'crop_pct': .9, 
        'interpolation': 'bicubic',
    }


default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(
        crop_pct=1.0
    ),

}


class CSWinTransformer(nn.Cell):
    """
    Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2,2,6,2], split_size = [3,5,7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
  
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads

        self.stage1_conv_embed = PatchEmbedding(image_size = img_size, embed_dim = embed_dim, k_size = 7, stride = 4, in_chans = in_chans, padding = 2)

        curr_dim = embed_dim
        dpr = [x.item() for x in np.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.CellList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size//4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, swap=False if i % 2 == 0 else True)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*2, resolution = img_size//4)
        curr_dim = curr_dim*2
        self.stage2 = nn.CellList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size//8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i], norm_layer=norm_layer, swap=False if i % 2 == 0 else True)
            for i in range(depth[1])])
        
        self.merge2 = Merge_Block(curr_dim, curr_dim*2, resolution=img_size// 8)
        curr_dim = curr_dim*2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size//16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer, swap=False if i % 2 == 0 else True)
            for i in range(depth[2])])

        self.stage3 = nn.CellList(temp_stage3)
        
        self.merge3 = Merge_Block(curr_dim, curr_dim*2, resolution = img_size// 16)
        curr_dim = curr_dim*2
        self.stage4 = nn.CellList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size//32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1])+i], norm_layer=norm_layer, last_stage=True)
            for i in range(depth[-1])])
       
        self.norm = norm_layer((curr_dim,))
        self.avgpool = P.ReduceMean(keep_dims=False)

        # Classifier head
        self.head = nn.Dense(curr_dim, num_classes) if num_classes > 0 else Identity()
        self.init_weights()
  

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
        
    def init_weights(self):
        """
        init_weights
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer.initializer(initializer.TruncatedNormal(sigma=0.02),  
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            
            elif isinstance(cell, nn.Conv2d) or isinstance(cell, nn.Conv1d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(initializer.initializer(initializer.Normal(sigma=math.sqrt(2.0 / fan_out)),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                                                             
                if cell.bias is not None:
                    cell.bias.set_data(initializer.initializer(initializer.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))      

            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer.initializer(initializer.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(initializer.initializer(initializer.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def forward_features(self, x):

        x = self.stage1_conv_embed(x)     # stem conv
        for blk in self.stage1:           # stage 1
            x = blk(x)
        
        x = self.merge1(x)
        for blk in self.stage2:
            x = blk(x)

        x = self.merge2(x)
        for blk in self.stage3:
            x = blk(x)
        
        x = self.merge3(x)
        for blk in self.stage4:
            x = blk(x)
        
        x = self.norm(x)
        x = self.avgpool(x, 1)
        
        return x    

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def CSWin_64_12211_tiny_224(**kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[1,2,21,1],
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4., **kwargs)
    
    model.default_cfg = default_cfgs['cswin_224']
    return model


def CSWin_64_24322_small_224(**kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4., **kwargs)
    
    model.default_cfg = default_cfgs['cswin_224']
    return model

def CSWin_96_24322_base_224(**kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[4,8,16,32], mlp_ratio=4., **kwargs)
    
    model.default_cfg = default_cfgs['cswin_224']
    return model

def CSWin_144_24322_large_224(**kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[6,12,24,24], mlp_ratio=4., **kwargs)
    
    model.default_cfg = default_cfgs['cswin_224']
    return model

if __name__ == "__main__":
    from mindspore import context

    model = CSWin_64_24322_small_224()
    n_parameters = sum(ops.Size()(p) for p in model.get_parameters() if p.requires_grad)
    print("num params:", n_parameters)
    x = Tensor(np.ones([1, 3, 224,224]), mindspore.float32)
    y = model(x)
    print("x.shape: ",y.shape)

    