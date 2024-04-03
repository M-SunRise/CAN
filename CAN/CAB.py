import torch
import torch.nn as nn

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class ChannelAggregationFFN(nn.Module):
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = nn.Mish()

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CAB(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., center_size = 2, around_size = 2, alpha=0.5, down_ratio=1):
        super(CAB, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = CA(dim, num_heads=num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., center_size = center_size, around_size = around_size, alpha=alpha, down_ratio=down_ratio)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ffn = ChannelAggregationFFN(dim, dim*4, 3)

    def forward(self, x):
        shortcut = x
        x = shortcut + self.attn(self.norm1(x))
        shortcut = x
        x = shortcut + self.ffn(self.norm2(x))
        return x


class CA(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., center_size = 2, around_size = 2, alpha=0.5, down_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)
        self.dim = dim
        self.conv = nn.Conv2d(self.dim, self.dim, 1, groups=self.dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.center_proj = nn.Linear(dim, dim)
        self.center_num_heads = num_heads
        self.center_size=center_size

        if down_ratio != 1:
            self.around_q = nn.Linear(dim, dim, bias=qkv_bias)
            self.around_kv = nn.Linear(dim, dim*2, bias=qkv_bias)
            self.aconv = nn.Conv2d(dim, dim, kernel_size=down_ratio, stride=down_ratio)
            self.anorm = nn.BatchNorm2d(dim)
        else:
            self.around_qkv = nn.Linear(dim, dim*3, bias=qkv_bias)

        self.around_proj = nn.Linear(dim, dim)
        self.around_num_heads = num_heads
        self.around_size = around_size

        self.softmax = nn.Softmax(dim=-1)
        self.scale = qk_scale or head_dim ** -0.5
        self.alpha = alpha
        self.down_ratio = down_ratio

    def CentralAttention(self, x):
        B, H, W, C = x.shape
        _left = int((self.alpha*0.5) * H)
        _right = int((1-self.alpha*0.5) * H)
        _top = int((self.alpha*0.5) * W)
        _down = int((1-self.alpha*0.5) * W)
        x = x[:, _left:_right, _top:_down, :]
        # x = x[:, H//4:3*H//4, W//4:3*W//4, :]
        B, H, W, C = x.shape
        G = self.center_size
        x = x.reshape(B, H // G, G, W // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B * H * W // G**2, G**2, C)

        #attention
        B_, N_, C_ = x.shape
        qkv = self.qkv(x).reshape(B_, N_, 3, self.center_num_heads, C_ // self.center_num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B, self.num_heads, N, N), N = H*W
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N_, C_)
        x = self.center_proj(x)
        x = x = x.reshape(B, H // G, W // G, G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous() # B, Hp//G, G, Wp//G, G, C
        x = x.reshape(B, H, W, C)
        return x


    def GlobalAttention(self, x):
        if self.down_ratio != 1:
            G = self.around_size
            g = (int)(self.around_size//self.down_ratio)
            around_x = self.anorm(self.aconv(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
            B, H, W, C = around_x.shape
            around_x = around_x.reshape(B, H // g, g, W // g, g, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            around_x = around_x.reshape(B * H * W // g**2, g**2, C)
            a_B, a_N, a_C = around_x.shape
            
            B, H, W, C = x.shape
            x = x.reshape(B, H // G, G, W // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.reshape(B * H * W // G**2, G**2, C)
            #attention
            B_, N_, C_ = x.shape
            q = self.around_q(x).reshape(B_, N_, 1, self.around_num_heads, C_ // self.around_num_heads).permute(2, 0, 3, 1, 4).contiguous()
            q = q[0]
            kv = self.around_kv(around_x).reshape(a_B, a_N, 2, self.around_num_heads, a_C // self.around_num_heads).permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1]
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)) # (B, self.num_heads, N, N), N = H*W
            attn = self.softmax(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N_, C_)
            x = self.around_proj(x)

            x = x = x.reshape(B, H // G, W // G, G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous() # B, Hp//G, G, Wp//G, G, C
            x = x.reshape(B, H, W, C)
        else:
            B, H, W, C = x.shape
            G = self.around_size
            x = x.reshape(B, H // G, G, W // G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x = x.reshape(B * H * W // G**2, G**2, C)

            #attention
            B_, N_, C_ = x.shape
            qkv = self.around_qkv(x).reshape(B_, N_, 3, self.around_num_heads, C_ // self.around_num_heads).permute(2, 0, 3, 1, 4).contiguous()
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1)) # (B, self.num_heads, N, N), N = H*W
            attn = self.softmax(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N_, C_)
            x = self.around_proj(x)

            x = x = x.reshape(B, H // G, W // G, G, G, C).permute(0, 1, 3, 2, 4, 5).contiguous() # B, Hp//G, G, Wp//G, G, C
            x = x.reshape(B, H, W, C)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        globalA = self.GlobalAttention(x)
        centralA = self.CentralAttention(x)
        _left = int((self.alpha*0.5) * H)
        _right = int((1-self.alpha*0.5) * H)
        _top = int((self.alpha*0.5) * W)
        _down = int((1-self.alpha*0.5) * W)
        globalA[:, _left:_right, _top:_down, :] = centralA
        x = globalA.permute(0, 3, 1, 2)
        return x
