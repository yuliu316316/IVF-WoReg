import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FB2(nn.Module):
    def __init__(self, embed=64):
        super(FB2, self).__init__()
        self.embed = embed
        self.feature_extract1 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed * 2, out_channels=self.embed * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embed * 2, out_channels=self.embed * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embed * 2, out_channels=self.embed, kernel_size=1, stride=1),
            nn.ReLU())

    def forward(self, x1, x2):
        out = self.feature_extract1(torch.cat((x1, x2), 1))
        return out


class FB3(nn.Module):
    def __init__(self, embed=64):
        super(FB3, self).__init__()
        self.embed = embed
        self.feature_extract1 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed * 3, out_channels=self.embed * 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embed * 3, out_channels=self.embed * 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embed * 3, out_channels=self.embed, kernel_size=1, stride=1),
            nn.ReLU())

    def forward(self, x1, x2, x3):
        out = self.feature_extract1(torch.cat((x1, x2, x3), 1))
        return out


class FB4(nn.Module):
    def __init__(self, embed=64):
        super(FB4, self).__init__()
        self.embed = embed
        self.feature_extract1 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed * 4, out_channels=self.embed * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embed * 4, out_channels=self.embed * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embed * 4, out_channels=self.embed, kernel_size=1, stride=1),
            nn.ReLU())

    def forward(self, x1, x2, x3, x4):
        out = self.feature_extract1(torch.cat((x1, x2, x3, x4), 1))
        return out


class FB6(nn.Module):
    def __init__(self, embed=64):
        super(FB6, self).__init__()
        self.embed = embed
        self.feature_extract1 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed * 6, out_channels=self.embed * 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embed * 6, out_channels=self.embed * 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embed * 6, out_channels=self.embed, kernel_size=1, stride=1),
            nn.ReLU())

    def forward(self, x1, x2, x3, x4, x5, x6):
        out = self.feature_extract1(torch.cat((x1, x2, x3, x4, x5, x6), 1))
        return out


class FusionLayer(nn.Module):
    def __init__(self, embedd=64):
        super(FusionLayer, self).__init__()
        self.Fb1 = FB2()
        self.Fb2 = FB4()
        self.Fb3 = FB3()
        self.Fb4 = FB4()
        self.Fb5 = FB6()
        self.Fb6 = FB4()
        self.embedd = embedd
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.embedd * 2, out_channels=self.embedd * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embedd * 2, out_channels=self.embedd * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.embedd * 2, out_channels=3, kernel_size=1, stride=1),
            nn.ReLU())

    def forward(self, x1, x2):
        out1 = self.Fb1(x1[0], x1[1])
        out2 = self.Fb2(x1[0], x1[1], x1[2], out1)
        out3 = self.Fb3(x1[1], x1[2], out2)
        out4 = self.Fb4(x2[0], x2[1], out1, out2)
        out5 = self.Fb5(x2[0], x2[1], x2[2], out2, out3, out4)
        out6 = self.Fb6(x2[1], x2[2], out3, out5)
        out = self.conv(torch.cat((out3, out6), 1))
        return out


class Aware_offset(nn.Module):
    def __init__(self, emded=64):
        super(Aware_offset, self).__init__()
        self.embed = emded
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed, out_channels=self.embed, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed, out_channels=self.embed, kernel_size=5, stride=1, padding=2),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed, out_channels=self.embed, kernel_size=7, stride=1, padding=3),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed * 3, out_channels=self.embed, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed * 2, out_channels=self.embed * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed * 2, out_channels=self.embed * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Sigmoid())
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed * 2, out_channels=self.embed, kernel_size=1, stride=1),
            nn.ReLU())
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=self.embed * 2, out_channels=self.embed, kernel_size=1, stride=1),
            nn.ReLU())

    def forward(self, x1, x2):  #
        in_x1 = self.conv1(x1)
        in_x2 = self.conv2(x1)
        in_x3 = self.conv3(x1)
        x = self.conv4(torch.cat((in_x1, in_x2, in_x3), 1))
        bottle_x1 = self.conv5(torch.cat((x1, x2), 1))
        bottle_x2 = self.conv6(bottle_x1)
        bottle_x1 = torch.mul(bottle_x1, bottle_x2)
        bottle_x1 = self.conv7(bottle_x1)
        x = torch.mul(x, bottle_x1)
        bottle_x3 = self.conv8(torch.cat((x1, x2), 1))
        x = torch.add(x, bottle_x3)
        return x


class Fe_extract(nn.Module):
    def __init__(self):
        super(Fe_extract, self).__init__()

        self.feature_extract1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

    def forward(self, x):
        out1 = self.feature_extract1(x)
        return out1


class one_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernal_size=3):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernal_size,
                              padding=kernal_size >> 1, stride=1)
        self.relu = nn.ReLU()
        self.normal = nn.LayerNorm(64)

    def forward(self, x):
        x1 = self.conv(x)
        _, _, H, W = x.shape
        x1 = x1.flatten(2).transpose(1, 2)
        B, L, C = x1.shape
        x1 = self.normal(x1)
        x1 = x1.view(B, H, W, C).permute(0, 3, 1, 2)
        output = self.relu(x1)
        return torch.cat((x, output), 1)


class RDB(nn.Module):
    def __init__(self, G0, C, G, kernaosize=3):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0 + i * G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0 + C * G, G, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class WindowAttention_correct(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.correct = nn.Linear(head_dim, 196, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x2):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_correct = x2.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x_correct = self.correct(x_correct)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + x_correct
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class StartTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.convqkv = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.convqkv(x)
        x = x.permute(0, 2, 3, 1)

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = shifted_x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # x1 = x.transpose(1, 2).view(B, C, H, W)

        return x


class SwinTransformerBlock_down8(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.conv1 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
                                   )
        self.convQKV1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)

    def forward(self, x, in_up, in_bottle):
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.convQKV1(x)
        x = x.permute(0, 2, 3, 1)
        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = shifted_x.view(B, H * W, C)
        x4 = shortcut + x
        x5 = x4.view(B, H, W, C).permute(0, 3, 1, 2)
        x = torch.cat((x5, in_bottle), 1)
        x = self.conv1(x)
        x = torch.add(x5, x)
        x = torch.cat((x, in_up), 1)
        x = self.conv2(x)
        x1 = torch.add(x5, x)

        x = x1.flatten(2).transpose(1, 2)

        # FFN

        x = x4 + self.mlp(self.norm2(x))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        return x1, x


class SwinTransformerBlock_correct(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim=64, num_heads=4, window_size=14,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_correct(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.aware_offset = Aware_offset()
        self.convQKV1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)

    def forward(self, x, x_basic):
        _, _, H, W = x.shape
        x_basic = self.aware_offset(x, x_basic)
        x_basic = x_basic.permute(0, 2, 3, 1)
        x = x.flatten(2).transpose(1, 2)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.convQKV1(x)
        x = x.permute(0, 2, 3, 1)
        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        x_basic = window_partition(x_basic, self.window_size)
        x_basic = x_basic.view(-1, self.window_size * self.window_size, C)
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, x_basic)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = shifted_x.view(B, H * W, C)
        # FFN
        x = x + self.mlp(self.norm2(x))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x


class BasicLayerbefore(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.RDB = RDB(64, 3, 64)
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_down8(dim=dim,
                                       num_heads=num_heads, window_size=window_size,

                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop, attn_drop=attn_drop,
                                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                       norm_layer=norm_layer)

            for i in range(depth)])

    def forward(self, x, in_up, in_bottle):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x1 = self.RDB(x)

                x2, x3 = blk(in_up, x1, in_bottle)

        return x1, x2, x3


class BasicLayer1(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.RDB = RDB(128, 3, 64)
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_down8(dim=dim,
                                       num_heads=num_heads, window_size=window_size,

                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale,
                                       drop=drop, attn_drop=attn_drop,
                                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                       norm_layer=norm_layer)

            for i in range(depth)])

    def forward(self, x, in_up, in_bottle, x_layer):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x1 = self.RDB(x)
                x5 = torch.add(x1, in_up)
                x2, x3 = blk(in_bottle, x5, x_layer)

        return x5, x2, x3


class SwinTransformer_in(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, depths=[1, 1], in_chans=3,
                 embed_dim=64, num_heads=[4, 4, 4],  #
                 window_size=7, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm  # true
        self.mlp_ratio = mlp_ratio
        self.num_layers = len(depths)
        self.Fe_extractconv = Fe_extract()
        self.compress_in = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer1(dim=int(embed_dim),

                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                norm_layer=norm_layer,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.layer_first = StartTransformerBlock(dim=int(embed_dim),
                                                 num_heads=4,
                                                 window_size=7,
                                                 mlp_ratio=2., qkv_bias=True,
                                                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                                                 act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.BasiclayerBefore = BasicLayerbefore(dim=int(embed_dim),

                                                 depth=depths[1],
                                                 num_heads=num_heads[0],
                                                 window_size=window_size,
                                                 mlp_ratio=self.mlp_ratio,
                                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                                 norm_layer=norm_layer,
                                                 use_checkpoint=use_checkpoint)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x1):
        conv_in = self.Fe_extractconv(x1)

        _, _, H, W = x1.shape
        x_up = self.layer_first(conv_in)

        B, L, C = x_up.shape
        x_up = x_up.transpose(1, 2).view(B, C, H, W)

        g_1 = []
        g_3 = []
        g_4 = []

        a1, a2, a3 = self.BasiclayerBefore(conv_in, x_up, x_up)
        g_1.append(a1)
        g_3.append(a3)
        g_4.append(a2)
        for i in range(len(self.layers)):
            x = torch.cat((g_3[i], g_1[i]), 1)
            b1, b2, b3 = self.layers[i](x, g_1[i], g_3[i], g_4[i])
            g_1.append(b1)
            g_4.append(b2)
            g_3.append(b3)
        return g_1, g_3


class Net(nn.Module):
    def __init__(self, num_layer=3):
        super(Net, self).__init__()
        self.up = SwinTransformer_in()
        self.bottle = SwinTransformer_in()
        self.Fusionlayer = FusionLayer()
        self.layer = nn.ModuleList()
        self.num_layer = num_layer
        for i_layer in range(self.num_layer):
            layer = SwinTransformerBlock_correct()
            self.layer.append(layer)

    def forward(self, x1, x2):
        up_g_1, up_g_3 = self.up(x1)
        bottle_g_1, bottle_g_3 = self.bottle(x2)
        correct_g_3 = []

        for iii in range(self.num_layer):
            x = self.layer[iii](up_g_3[iii], bottle_g_1[iii])
            correct_g_3.append(x)
        out = self.Fusionlayer(bottle_g_3, correct_g_3)
        return out, correct_g_3
