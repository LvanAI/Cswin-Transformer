from mindspore import nn, Tensor
from mindspore.ops import operations as ops

from src.models.blocks.misc import Identity, DropPath1D


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp,self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features, has_bias=True)
        keep_rate = 1 - drop
        self.drop = nn.Dropout(keep_rate)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Cell):
    def __init__(self,dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(1 - attn_drop)

        self.proj = nn.Dense(dim, dim,  has_bias=True)
        self.proj_drop = nn.Dropout(1 - proj_drop)

        self.unstack = ops.Unstack(axis=0)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)
        self.mul = ops.Mul()
        self.attn_matmul_v = ops.BatchMatMul()


    def construct(self, x):
        B, N, C = x.shape
        
        qkv =  self.reshape(self.qkv(x), (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = self.unstack(qkv)

        attn = self.mul(self.q_matmul_k(q, k), self.scale) 

        attn = self.softmax(attn)
        attn = self.attn_drop(attn) 

        attn = self.reshape(self.transpose( self.attn_matmul_v(attn, v), (0, 2, 1, 3)), (B,N,C))

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,  drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,))

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x