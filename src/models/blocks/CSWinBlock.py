

import mindspore
from mindspore import nn, Tensor

from src.models.blocks.misc import Identity
from src.models.blocks.drop_path import DropPath
import mindspore.ops as ops

class PadConvd(nn.Cell):
    def __init__(self, dim):
        super(PadConvd,self).__init__()
        
        self.convd = nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, pad_mode = "valid", has_bias = True, group = dim)
        self.pad = nn.Pad(paddings=((0, 0),(0, 0),(1, 1), (1, 1)), mode="CONSTANT")

    def construct(self, x):
        x = self.pad(x)
        x = self.convd(x)

        return x

class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob = 1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Cell):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super(LePEAttention, self).__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or Tensor(head_dim ** -0.5, mindspore.float32)
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)

        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.split_size == 1:
            self.get_v = nn.Conv1d(dim, dim, kernel_size = 3, stride = 1, pad_mode = "pad", padding = 1, has_bias=True, group = dim)
            #self.get_vw = nn.Conv1d(dim, dim, kernel_size = 3, stride = 1, pad_mode = "pad", padding = 1, has_bias=True, group = dim)
        else:
            self.get_v = nn.Conv2d(dim, dim, kernel_size = 3, stride = 1, pad_mode = "pad", padding = 1, has_bias=True, group = dim)

        self.attn_drop = nn.Dropout(keep_prob = 1.0 - attn_drop)   
        

    def img2windows(self,img, H_sp, W_sp):
        """
        img: B C H W
        """
        B, C, H, W = img.shape
        
        img_reshape = ops.Reshape()(img, (B, C, H // H_sp, H_sp, W // W_sp, W_sp))
        img_out = ops.Reshape()(ops.Transpose()(img_reshape, (0, 2, 4, 3, 5, 1)), (-1, H_sp* W_sp, C))
        return img_out


    def windows2img(self,B, img_splits_hw, H_sp, W_sp, H, W):
        """
        img_splits_hw: B' H W C
        """
        img = ops.Reshape()(img_splits_hw, (B, H // H_sp, W // W_sp, H_sp, W_sp, -1))
        img = ops.Reshape()(ops.Transpose()(img, (0, 1, 3, 2, 4, 5)), (B, H, W, -1))
        return img_splits_hw
    
    def im2cswin(self, x):
        B, _, C = x.shape
        H = W = self.resolution
        #x = ops.Reshape()(ops.Transpose()(x, (0, 2, 1)), (B, C, H, W))  
        #x = self.img2windows(x, self.H_sp, self.W_sp)
        x = ops.Transpose()(ops.Reshape()(x, (B, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp, C)), (0, 1, 3, 2, 4, 5))
        if self.num_heads == 1:
            x = ops.Reshape()(x, (-1, 1, self.H_sp*self.W_sp, C))
        else:
            x = ops.Transpose()(ops.Reshape()(x, (-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)), (0, 2, 1, 3))
        return x

    def get_lepe(self, x):
        B, _, C = x.shape
        H = W = self.resolution
        #x = ops.Reshape()(ops.Transpose()(x,(0, 2, 1)), (B, C, H, W))

        H_sp, W_sp = self.H_sp, self.W_sp
        #x = ops.Reshape()(x , (B, C, H // H_sp, H_sp, W // W_sp, W_sp))
        #x = ops.Reshape()(ops.Transpose()(x, (0, 2, 4, 1, 3, 5)), (-1, C, H_sp, W_sp))
        x = ops.Reshape()(x, (B, H // H_sp, H_sp, W // W_sp, W_sp, C))
        x = ops.Reshape()(ops.Transpose()(x, (0, 1, 3, 5, 2, 4)), (-1, C, H_sp, W_sp))
        if H_sp == 1:
            x = ops.Reshape()(x, (-1, C, W_sp))
            lepe = self.get_v(x)
            lepe = ops.Reshape()(lepe, (-1, C, H_sp, W_sp))
        elif W_sp == 1:
            x = ops.Reshape()(x, (-1, C, H_sp))
            lepe = self.get_v(x)
            lepe = ops.Reshape()(lepe, (-1, C, H_sp, W_sp))
        else:
            lepe = self.get_v(x) ### B', C, H', W'

        lepe = ops.Transpose()(ops.Reshape()(lepe, (-1, self.num_heads, C // self.num_heads, H_sp * W_sp)),  (0, 1, 3, 2))
        x = ops.Transpose()(ops.Reshape()(x, (-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp)),  (0, 1, 3, 2))

        return x, lepe

    def construct(self, q, k, v):
        """
        x: B L C
        """
        ### Img2Window
        H = W = self.resolution
        B, _, C = q.shape

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v)

        q = ops.Mul()(q, self.scale)
        attn = ops.BatchMatMul(transpose_b = True)(q, k)
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)

        x = ops.BatchMatMul()(attn, v) + lepe
        #x = ops.Reshape()(ops.Transpose()(x, (0, 2, 1, 3)), (-1, self.H_sp* self.W_sp, C))   # B head N N @ B head N C
        if self.num_heads == 1:
            x = ops.Reshape()(x, (B, H // self.H_sp, W // self.W_sp, self.H_sp, self.W_sp, C))
            x = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))
        else:
            x = ops.Reshape()(x, (B, H // self.H_sp, W // self.W_sp, self.num_heads, self.H_sp, self.W_sp, C // self.num_heads))
            x = ops.Transpose()(x, (0, 1, 4, 2, 5, 3, 6))

        ### Window2Img
        #x = self.windows2img(B, x, self.H_sp, self.W_sp, H, W)  # B H' W' C
        x = ops.Reshape()(x, (B, -1, C))
        return x


class CSWinBlock(nn.Cell):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False, swap=False):
        super(CSWinBlock, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2

        if last_stage: 
            self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
            self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
            self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        else:
            self.q1 = nn.Dense(in_channels=dim, out_channels=dim//2, has_bias=qkv_bias)
            self.q2 = nn.Dense(in_channels=dim, out_channels=dim//2, has_bias=qkv_bias)
            self.k1 = nn.Dense(in_channels=dim, out_channels=dim//2, has_bias=qkv_bias)
            self.k2 = nn.Dense(in_channels=dim, out_channels=dim//2, has_bias=qkv_bias)
            self.v1 = nn.Dense(in_channels=dim, out_channels=dim//2, has_bias=qkv_bias)
            self.v2 = nn.Dense(in_channels=dim, out_channels=dim//2, has_bias=qkv_bias)

        self.norm1 = norm_layer((dim,))

        self.proj = nn.Dense(dim, dim, has_bias=True)
        self.proj_drop = nn.Dropout(keep_prob = 1.0 - drop)
        
        if last_stage:
            self.attns = nn.CellList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            if swap:
                idxs = [self.branch_num-1-i for i in range(self.branch_num)]
            else:
                idxs = [i for i in range(self.branch_num)]
            self.attns = nn.CellList([
                LePEAttention(
                    dim//2, resolution=self.patches_resolution, idx = idxs[i],
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer((dim,))


    def construct(self, x):
        """
        x: B, H*W, C
        """
        B, _, C = x.shape
        img = self.norm1(x)

        if self.branch_num == 2:
            q1, q2 = self.q1(img), self.q2(img)
            k1, k2 = self.k1(img), self.k2(img)
            v1, v2 = self.v1(img), self.v2(img)
            x1 = self.attns[0](q1, k1, v1)
            x2 = self.attns[1](q2, k2, v2)
            attened_x = ops.Concat(axis=2)((x1,x2))
        else:
            q = self.q(img)
            k = self.k(img)
            v = self.v(img)
            attened_x = self.attns[0](q,k,v)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Merge_Block(nn.Cell):
    def __init__(self, dim, dim_out, resolution, norm_layer=nn.LayerNorm):
        super(Merge_Block, self).__init__()
        self.conv = nn.Conv2d(dim, dim_out, kernel_size = 3, stride = 2, pad_mode = "pad", padding = 1, has_bias=True)
        self.norm = norm_layer((dim_out,))
        self.resolution = resolution

    def construct(self, x):
        B, _, C = x.shape
        H = W =  self.resolution

        x = ops.Reshape()(ops.Transpose()(x, (0,2,1)), (B, C, H, W))
        x = self.conv(x)
        
        B, C = x.shape[:2]
        x = ops.Transpose()(ops.Reshape()(x, (B, C, -1)), (0,2,1))
        x = self.norm(x)
        return x
