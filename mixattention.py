class MixingAttention(nn.Layer):
    r""" Mixing Attention Module.
    Modified from Window based multi-head self attention (W-MSA) module
    with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        dwconv_kernel_size (int): The kernel size for dw-conv
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to
            query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale
            of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """
    def __init__(self,
                 dim,
                 window_size,
                 dwconv_kernel_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        attn_dim = dim // 2
        self.window_size = window_size  # Wh, Ww
        self.dwconv_kernel_size = dwconv_kernel_size
        self.num_heads = num_heads
        head_dim = attn_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = self.create_parameter(
            shape=((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                   num_heads),
            default_initializer=zeros_)
        self.add_parameter("relative_position_bias_table",
                           self.relative_position_bias_table)

        # get pair-wise relative position index for each token
        # inside the window
        relative_coords = self._get_rel_pos()
        self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             self.relative_position_index)
        # prev proj layer
        self.proj_attn = nn.Linear(dim, dim // 2)
        self.proj_attn_norm = nn.LayerNorm(dim // 2)
        self.proj_cnn = nn.Linear(dim, dim)
        self.proj_cnn_norm = nn.LayerNorm(dim)

        # conv branch
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2D(
                dim, dim,
                kernel_size=self.dwconv_kernel_size,
                padding=self.dwconv_kernel_size // 2,
                groups=dim
            ),
            nn.BatchNorm(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.Conv2D(dim, dim // 8, kernel_size=1),
            nn.BatchNorm(dim // 8),
            nn.GELU(),
            nn.Conv2D(dim // 8, dim // 2, kernel_size=1),
        )
        self.projection = nn.Conv2D(dim, dim // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm(dim // 2)

        # window-attention branch
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.spatial_interaction = nn.Sequential(
            nn.Conv2D(dim // 2, dim // 16, kernel_size=1),
            nn.BatchNorm(dim // 16),
            nn.GELU(),
            nn.Conv2D(dim // 16, 1, kernel_size=1)
        )
        self.attn_norm = nn.LayerNorm(dim // 2)

        # final projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table)
        self.softmax = nn.Softmax(axis=-1)

    def _get_rel_pos(self):
        """ Get pair-wise relative position index for each token inside the window.
        Args:
            window_size (tuple[int]): window size
        """
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        # 2, Wh, Ww
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
        coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
        relative_coords = coords_flatten_1 - coords_flatten_2
        relative_coords = relative_coords.transpose(
            [1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[
            0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        return relative_coords

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            H: the height of the feature map
            W: the width of the feature map
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww)
                or None
        """
        # B * H // win * W // win x win*win x C
        x_atten = self.proj_attn_norm(self.proj_attn(x))
        x_cnn = self.proj_cnn_norm(self.proj_cnn(x))
        # B * H // win * W // win x win*win x C --> B, C, H, W
        x_cnn = window_reverse2(x_cnn, self.window_size, H, W, x_cnn.shape[-1])

        # conv branch
        x_cnn = self.dwconv3x3(x_cnn)
        channel_interaction = self.channel_interaction(
            F.adaptive_avg_pool2d(x_cnn, output_size=1))
        x_cnn = self.projection(x_cnn)

        # attention branch
        B_, N, C = x_atten.shape
        qkv = self.qkv(x_atten).reshape(
            [B_, N, 3, self.num_heads, C // self.num_heads]).transpose(
                [2, 0, 3, 1, 4])
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # channel interaction
        x_cnn2v = F.sigmoid(channel_interaction).reshape(
            [-1, 1, self.num_heads, 1, C // self.num_heads])
        v = v.reshape(
            [x_cnn2v.shape[0], -1, self.num_heads, N, C // self.num_heads])
        v = v * x_cnn2v
        v = v.reshape([-1, self.num_heads, N, C // self.num_heads])

        q = q * self.scale
        attn = paddle.mm(q, k.transpose([0, 1, 3, 2]))

        index = self.relative_position_index.reshape([-1])

        relative_position_bias = paddle.index_select(
            self.relative_position_bias_table, index)
        relative_position_bias = relative_position_bias.reshape([
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1
        ])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose(
            [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N]) + \
                mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x_atten = paddle.mm(attn, v).transpose([0, 2, 1, 3]).reshape(
            [B_, N, C])

        # spatial interaction
        x_spatial = window_reverse2(x_atten, self.window_size, H, W, C)
        spatial_interaction = self.spatial_interaction(x_spatial)
        x_cnn = F.sigmoid(spatial_interaction) * x_cnn
        x_cnn = self.conv_norm(x_cnn)
        # B, C, H, W --> B * H // win * W // win x win*win x C
        x_cnn = window_partition2(x_cnn, self.window_size)

        # concat
        x_atten = self.attn_norm(x_atten)
        x = paddle.concat([x_atten, x_cnn], axis=-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x