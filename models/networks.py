import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
from pytorch_msssim import ssim, ms_ssim
from timm.models.layers import DropPath
import yaml
from yaml import CLoader


###############################################################################
# Helper Functions
###############################################################################


def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(m.bias)


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "syncbatch":
        norm_layer = functools.partial(nn.SyncBatchNorm, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                "BatchNorm2d") != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type="normal", init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    import os

    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            net.to(local_rank)
            print(f"Initialized with device cuda:{local_rank}")
        else:
            net.to(0)
            print("Initialized with device cuda:0")
    init_weights(net, init_type, init_gain=init_gain)
    return net


class WindowedCrossAttentionBlock(nn.Module):
    """Pre-LN block with windowed multi-head self-attention and FiLM conditioning.

    Two design choices target the gradient-starvation observed in the prior
    `UltraLightPatchCrossAttentionBlock`:

    1. Replace per-patch scalar sigmoid gate with multi-head softmax attention
       inside non-overlapping windows. Each block then performs real spatial
       reasoning across patches — no longer an identity-like passthrough.
    2. Inject the conditioning signal via FiLM (per-channel scale + shift) on
       the LayerNorm'd input. This gives `cond_patch` a direct gradient path
       into every block, rather than the previous bottleneck through one
       scalar dot-product gate.

    LayerScale init at 1.0 keeps both residual branches live from step 1.
    FiLM weights init at 0 → block starts as pure self-attention; conditioning
    influence ramps up as gradients update the FiLM projection.
    """

    def __init__(self, dim, num_heads=4, window_size=8, mlp_ratio=2.0, drop_path=0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.film_norm = nn.LayerNorm(dim)
        self.film = nn.Linear(dim, dim * 2)
        # Small random init (not zero) so gradient flows from x's path back
        # through cond → pid_encoder from step 1. With std=0.02 and fan_in=dim,
        # initial (scale, shift) is ~N(0, 0.02²·dim) → mild ~±0.16 modulation;
        # block stays close to identity-attention but cond is wired up.
        nn.init.normal_(self.film.weight, std=0.02)
        nn.init.zeros_(self.film.bias)

        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

        self.layer_scale_1 = nn.Parameter(torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, cond, ny, nx):
        # x:    [B, N, C], N = ny*nx (row-major)
        # cond: [B, N, C]
        scale, shift = self.film(self.film_norm(cond)).chunk(2, dim=-1)
        x_norm = self.norm1(x) * (1.0 + scale) + shift
        x_attn = self._windowed_attention(x_norm, ny, nx)
        x = x + self.drop_path(self.layer_scale_1 * x_attn)
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x

    def _windowed_attention(self, x, ny, nx):
        ws = self.window_size
        B, N, C = x.shape
        pad_h = (ws - ny % ws) % ws
        pad_w = (ws - nx % ws) % ws
        x_2d = x.view(B, ny, nx, C)
        if pad_h or pad_w:
            x_2d = F.pad(x_2d, (0, 0, 0, pad_w, 0, pad_h))
        ny_p, nx_p = ny + pad_h, nx + pad_w

        x_w = (
            x_2d.view(B, ny_p // ws, ws, nx_p // ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5).contiguous()
            .view(-1, ws * ws, C)
        )

        Bw, Nw, _ = x_w.shape
        qkv = (
            self.qkv(x_w)
            .view(Bw, Nw, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(Bw, Nw, C)
        out = self.proj(out)

        out = (
            out.view(B, ny_p // ws, nx_p // ws, ws, ws, C)
            .permute(0, 1, 3, 2, 4, 5).contiguous()
            .view(B, ny_p, nx_p, C)
        )
        if pad_h or pad_w:
            out = out[:, :ny, :nx, :]
        return out.reshape(B, N, C)


class ResidualUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, upsample=False):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample

        stride = 2 if self.downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3,
                               padding_mode='reflect')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, padding='same', padding_mode='reflect')
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                              stride=stride,
                              padding=0) if in_channels != out_channels or self.downsample else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        out = self.relu(x + identity)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2.0, mode='bilinear', align_corners=False)
        return out


class UltraLightPatchCrossAttentionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0, drop_path=0.2):
        super().__init__()
        # Patch feature → Q, K, V (but Q, K = x_patch, V = concat[x_patch, cond])
        self.norm = nn.LayerNorm(dim)
        self.qk = nn.Linear(dim, dim * 2, bias=True)
        self.v_cond = nn.Linear(dim * 2, dim, bias=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma = nn.Parameter(1e-4 * torch.ones(dim))  # LayerScale

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(0.2),
        )
        self.qk.apply(init_weights_kaiming)
        self.v_cond.apply(init_weights_kaiming)
        self.mlp.apply(init_weights_kaiming)

    def forward(self, x_patch, cond_patch):
        # x_patch: [B, N, C]
        # cond_patch: [B, N, cond_dim]  (PID + pos%)
        B, N, C = x_patch.shape
        norm_x = self.norm(x_patch)
        qk = self.qk(norm_x)  # [B, N, 2C]
        q, k = qk.chunk(2, dim=-1)  # [B, N, C], [B, N, C]
        # Attention scores per patch (so this is just an elementwise gating)
        attn = (q * k).sum(-1, keepdim=True) / math.sqrt(C)  # [B, N, 1]
        attn = attn.sigmoid()  # Soft gating, [0,1]

        v_input = torch.cat([x_patch, cond_patch], dim=-1)
        v = self.v_cond(v_input)  # 3364x256 and 192x128
        out = x_patch + self.drop_path(self.gamma * attn * v)
        out = out + self.drop_path(self.mlp(out))
        return out


class PIDConvWithRoPE(nn.Module):
    def __init__(self, input_channels=2, output_dim=64, seq_len=400):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        # Conv1D: (B, C_in, N) → (B, C_out, N)
        self.project = nn.Conv1d(in_channels=input_channels, out_channels=output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, 2, N] or [B*N, 2, W]
        x = self.project(x)  # [B, C, N] or [B*N, C, W]
        x = x.permute(0, 2, 1)  # [B, N, C] or [B*N, W, C]
        x = self.apply_rope(x)  # [B, N, C] or [B*N, W, C]
        x = x.mean(dim=1)  # [B, C] or [B*N, C]
        return x

    def apply_rope(self, x):
        # RoPE on last dim (C)
        B, N, C = x.size()
        assert C % 2 == 0, "C must be even for RoPE"

        # Positional indices [0...N-1]
        pos = torch.arange(N, device=x.device).unsqueeze(1)  # [N, 1]
        dim = torch.arange(C // 2, device=x.device).unsqueeze(0)  # [1, C//2]

        theta = 1000 ** (-2 * dim / C)
        angles = pos * theta  # [N, C//2]

        sin_embed = torch.sin(angles).unsqueeze(0).repeat(B, 1, 1)  # [B, N, C//2]
        cos_embed = torch.cos(angles).unsqueeze(0).repeat(B, 1, 1)

        x1, x2 = x[..., ::2], x[..., 1::2]  # Interleaved split
        x_rope = torch.cat([x1 * cos_embed - x2 * sin_embed,
                            x1 * sin_embed + x2 * cos_embed], dim=-1)
        return x_rope


class DynamicPatchEmbedWithRoPE(nn.Module):
    def __init__(self, patch_size=9, in_chans=1, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size * patch_size * in_chans, embed_dim)

    def forward(self, x):
        # x: [B, 1, H, W]
        # pid_signal: [B, 2, H, SimColSize] (optional, else returns None)
        pad = self.patch_size // 2
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='replicate')
        unfold = torch.nn.Unfold(kernel_size=self.patch_size, stride=1)
        patches = unfold(x)  # [B, C*K*K, N_patches]
        patches = patches.transpose(1, 2)  # [B, N_patches, C*K*K]
        x_embed = self.proj(patches)  # [B, N_patches, embed_dim]
        x_embed = self.apply_rope(x_embed)  # [B, N_patches, embed_dim]

        # --- Patch position info for PID/pos% ---
        B, N_patches, _ = x_embed.shape
        H, W = x.shape[2], x.shape[3]
        stride = 1
        n_x = (W - self.patch_size) // stride + 1  # == orig W
        n_y = (H - self.patch_size) // stride + 1  # == orig H
        # Vectorized center positions (patch_cols/rows are original-space indices 0..n-1)
        patch_rows = torch.arange(n_y, device=x.device).repeat_interleave(n_x)
        patch_cols = torch.arange(n_x, device=x.device).repeat(n_y)
        y_centers = patch_rows * stride + pad  # padded-space row index

        # Broadcast to batch dimension
        y_centers = y_centers.unsqueeze(0).expand(B, -1)  # [B, N_patches]
        pos_percents = patch_cols.float() / float(max(n_x - 1, 1))  # [N_patches] 0..1
        pos_percents = pos_percents.unsqueeze(0).expand(B, -1).unsqueeze(-1)  # [B, N_patches, 1]
        y_idx = y_centers.clamp(max=H - 1).long()  # [B, N_patches]
        pid_inputs = []
        for b in range(B):
            rowvals = x[b, 0, y_idx[b], :]  # [N_patches, W]
            pid_inputs.append(rowvals)  # [N_patches, 1, W]
        pid_inputs = torch.stack(pid_inputs, dim=0)  # [B, N_patches, 1, W]
        return (x_embed, pid_inputs, pos_percents, n_x, n_y)

    def apply_rope(self, x):
        B, N, D = x.shape
        half_dim = D // 2
        freq_seq = torch.arange(half_dim, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))  # [half_dim]
        pos = torch.arange(N, dtype=torch.float32, device=x.device)  # [N]
        sinusoid = torch.einsum('n,d->nd', pos, inv_freq)  # [N, half_dim]
        sin = torch.sin(sinusoid)
        cos = torch.cos(sinusoid)
        sin = sin.unsqueeze(0).repeat(B, 1, 1)  # [B, N, half_dim]
        cos = cos.unsqueeze(0).repeat(B, 1, 1)
        x1, x2 = x[..., :half_dim], x[..., half_dim:]
        x_rotated = torch.cat([x1 * cos - x2 * sin,
                               x1 * sin + x2 * cos], dim=-1)
        return x_rotated


def reconstruct_from_patches(z_output, H_patch, W_patch, depth=32):
    # z_output: [B, N * Depth, ph, pw]
    B, N, ph, pw = z_output.shape
    assert N == H_patch * W_patch * depth, f"Patch 수 불일치: {N} vs {H_patch}x{W_patch}x{depth}"

    z = z_output.view(B, H_patch, W_patch, ph, pw, depth)
    z = z.permute(0, 5, 1, 3, 2, 4)  # [B, depth, H_patch, W_patch, ph, pw]
    z = z.contiguous().view(B, depth, H_patch * ph, W_patch * pw)
    return z


class PIDSwinModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        depth = config['model']['conv']['depth']
        patch_size = config['model']['patch_size']
        embed_dim = config['model']['embed_dim']
        self.depth = depth
        self.patch_embed = DynamicPatchEmbedWithRoPE(patch_size, 1, embed_dim)
        self.pid_encoder = PIDConvWithRoPE(input_channels=2, output_dim=embed_dim)
        self.blocks = nn.ModuleList([
            UltraLightPatchCrossAttentionBlock(embed_dim)
            for _ in range(config['model']['block_repeat'])
        ])
        self.upsample = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, 16 * depth),  # 4x4 patch = 16
        )
        if config['model']['conv']['embed']:
            self.out = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(depth, config['model']['conv']['depth'], kernel_size=config['model']['conv']['k_size'],
                          padding='same', padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                nn.Conv2d(config['model']['conv']['depth'], 1, kernel_size=config['model']['conv']['k_size'],
                          padding='same', padding_mode='reflect')
            )
            self.out.apply(init_weights_kaiming)
        else:
            self.out = nn.Identity()

    def forward(self, img):
        x, pid_patch, pos_patch, nx, ny = self.patch_embed(img)
        B, N, W = pid_patch.shape
        pos_patch_expand = pos_patch.expand(-1, -1, W)  # [B, N, W]
        cond_input = torch.cat([pid_patch, pos_patch_expand], dim=2)  # [B, 841, 128]
        cond_input = cond_input.unsqueeze(2)
        cond_input = cond_input.view(B * N, 2, W)
        cond_patch = self.pid_encoder(cond_input).view(B, N, -1)  # [B, N, cond_dim]
        for single in self.blocks:
            x = single(x, cond_patch)
        z_output = self.upsample(x)  # [B, N, 16]
        z_output = z_output.view(x.size(0), -1, 4, 4)
        z = reconstruct_from_patches(z_output, ny, nx, self.depth)
        return self.out(z)


def define_G(input_nc, output_nc, ngf, netG, norm="batch", use_dropout=False, init_type="normal", init_gain=0.02):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_128 | unet_256 | afm_optimized | sr_4x | down_4x
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == "resnet_9blocks" or netG == "afm_optimized":
        net = OptimizedAFMGenerator(input_nc, output_nc, ngf=ngf, n_blocks=9)
    elif netG == "sr_4x":
        # net = SRGenerator(input_nc, output_nc, ngf=ngf, n_blocks=9)
        filename = "/home/psdl/Workspace/HS-AFM-UPSCALER/config/node0/config.yaml"
        config = yaml.load(open(filename, "r").read(), CLoader)
        net = PIDSwinModel(config)
    elif netG == "down_4x":
        net = DownsampleGenerator(input_nc, output_nc, ngf=ngf, n_blocks=9)
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == "unet_128":
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == "unet_256":
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm="batch", init_type="normal", init_gain=0.02):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel | multiscale
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic" or netD == "multiscale":
        net = MultiscaleAFMDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    elif netD == "n_layers":  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == "pixel":  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    elif netD == "attn":  # classify if each pixel is real or fake
        net = AttnDiscriminator(input_nc, ndf)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netD)
    return net


def robust_log_l1_loss(pred, target):
    # sign(x) * log(1 + |x|) 적용
    # 물리 센서 데이터의 아웃라이어 영향을 로그 스케일로 감쇄
    s_pred = torch.sign(pred) * torch.log1p(torch.abs(pred))
    s_target = torch.sign(target) * torch.log1p(torch.abs(target))
    return F.mse_loss(s_pred, s_target)


##############################################################################
# Classes
##############################################################################
class MultiscaleAFMDiscriminator(nn.Module):
    """
    Multiscale Discriminator for AFM images.
    Uses multiple PatchGANs at different scales to capture both 
    fine-grained noise/texture and larger structural artifacts (like scan lines).
    Uses Spectral Normalization for stability.
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, num_D=2):
        super(MultiscaleAFMDiscriminator, self).__init__()
        self.num_D = num_D

        for i in range(num_D):
            netD = NLayerDiscriminatorOptimized(input_nc, ndf, n_layers, norm_layer)
            setattr(self, f"layer{i}", netD)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        input_downsampled = input
        for i in range(self.num_D):
            model = getattr(self, f"layer{i}")
            result.append(model(input_downsampled))
            if i < self.num_D - 1:
                input_downsampled = self.downsample(input_downsampled)
        return result


class NLayerDiscriminatorOptimized(nn.Module):
    """Optimized PatchGAN discriminator with Spectral Norm"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        super(NLayerDiscriminatorOptimized, self).__init__()

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def robust_log_l1_loss(pred, target):
    # sign(x) * log(1 + |x|) 적용
    # 물리 센서 데이터의 아웃라이어 영향을 로그 스케일로 감쇄
    s_pred = torch.sign(pred) * torch.log1p(torch.abs(pred))
    s_target = torch.sign(target) * torch.log1p(torch.abs(target))
    return F.l1_loss(s_pred, s_target)


class StructuralLoss(nn.Module):
    def __init__(self, alpha=0.84):
        super(StructuralLoss, self).__init__()
        self.alpha = alpha  # MS-SSIM과 L1/MSE 사이의 밸런스 가중치
        self.l1 = nn.L1Loss()

    def forward(self, img1, img2):
        img1, img2 = img1.float(), img2.float()
        # AFM 데이터 특성상 정규화가 중요함
        img1_norm = (torch.tanh(img1) + 1.0) / 2.0
        img2_norm = (torch.tanh(img2) + 1.0) / 2.0

        # ssim은 [0, 1] 범위에서 동작
        loss_ssim = 1 - ssim(img1_norm, img2_norm, data_range=1.0, size_average=True)
        loss_l1 = self.l1(img1, img2)
        return self.alpha * loss_ssim + (1 - self.alpha) * loss_l1


##############################################################################
# Classes
##############################################################################
class ECABlock(nn.Module):
    """Efficient Channel Attention module"""

    def __init__(self, channels, b=1, gamma=2):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ResBlockPlain(nn.Module):
    """Simple ResNet block without Spectral Norm or Attention to avoid artifacts."""

    def __init__(self, dim, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super(ResBlockPlain, self).__init__()
        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            norm_layer(dim)
        ]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class OptimizedAFMGenerator(nn.Module):
    """
    Smooth Generator for AFM images (Checkerboard-free).
    Uses Bilinear Upsampling + Conv instead of PixelShuffle.
    Avoids Spectral Norm and Attention in the Generator to prevent periodic artifacts.
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=64, n_blocks=9):
        super(OptimizedAFMGenerator, self).__init__()

        # Initial Stem
        self.begin = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Bottleneck
        blocks = []
        for _ in range(n_blocks):
            blocks += [ResBlockPlain(ngf * 4)]
        self.bottleneck = nn.Sequential(*blocks)

        # Decoder with Bilinear Upsampling + Skip Connections
        # up1: 1/4 -> 1/2
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4 + ngf * 2, ngf * 2, 3, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # up2: 1/2 -> 1
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2 + ngf, ngf, 3, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        # Final
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
        )

    def forward(self, x):
        # Encoder + Skips
        s0 = self.begin(x)
        s1 = self.down1(s0)
        x = self.down2(s1)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder + Skip Concat
        x = self.up1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.fuse1(x)

        x = self.up2(x)
        x = torch.cat([x, s0], dim=1)
        x = self.fuse2(x)

        return self.final(x)


class SRGenerator(nn.Module):
    """4x Super-Resolution Generator: LR(64x64) -> HR(256x256).
    Encoder-bottleneck-decoder (same as OptimizedAFMGenerator) followed by
    two additional bilinear upsample stages to reach 4x the input size.
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=64, n_blocks=9):
        super(SRGenerator, self).__init__()

        self.begin = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.bottleneck = nn.Sequential(*[ResBlockPlain(ngf * 4) for _ in range(n_blocks)])

        # Decoder back to LR size with skip connections
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4 + ngf * 2, ngf * 2, 3, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2 + ngf, ngf, 3, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        # SR upsample: LR size → 2x → 4x (no encoder skips here)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, 3, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf // 2, 3, bias=True),
            nn.InstanceNorm2d(ngf // 2),
            nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf // 2, out_channels, 7),
        )

    def forward(self, x):
        s0 = self.begin(x)
        s1 = self.down1(s0)
        x = self.down2(s1)
        x = self.bottleneck(x)

        x = self.up1(x)
        x = self.fuse1(torch.cat([x, s1], dim=1))
        x = self.up2(x)
        x = self.fuse2(torch.cat([x, s0], dim=1))

        x = self.up3(x)
        x = self.up4(x)
        return self.final(x)


class DownsampleGenerator(nn.Module):
    """4x Downscale Generator: HR(256x256) -> LR(64x64).
    Two stride-2 convolutions reduce the input by 4x before passing through
    the same encoder-bottleneck-decoder structure as OptimizedAFMGenerator,
    which outputs at the reduced (LR) size.
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=64, n_blocks=9):
        super(DownsampleGenerator, self).__init__()

        # 4x spatial reduction: 256 -> 128 -> 64
        self.pre_down1 = nn.Sequential(
            nn.Conv2d(in_channels, ngf // 2, 3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf // 2),
            nn.ReLU(True)
        )
        self.pre_down2 = nn.Sequential(
            nn.Conv2d(ngf // 2, ngf, 3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        # Same structure as OptimizedAFMGenerator from here (operates at LR size)
        self.begin = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, ngf, 7, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.bottleneck = nn.Sequential(*[ResBlockPlain(ngf * 4) for _ in range(n_blocks)])

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4 + ngf * 2, ngf * 2, 3, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2 + ngf, ngf, 3, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
        )

    def forward(self, x):
        x = self.pre_down1(x)
        x = self.pre_down2(x)

        s0 = self.begin(x)
        s1 = self.down1(s0)
        x = self.down2(s1)
        x = self.bottleneck(x)

        x = self.up1(x)
        x = self.fuse1(torch.cat([x, s1], dim=1))
        x = self.up2(x)
        x = self.fuse2(torch.cat([x, s0], dim=1))
        return self.final(x)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = F.mse_loss
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == "real":  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError(f"{type} not implemented")
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True,
                                        retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type="reflect"):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ChannelGate(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(c, c // r, bias=False), nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(B, C)
        mx = torch.max(torch.max(x, dim=3, keepdim=True)[0], dim=2, keepdim=True)[0].view(B, C)
        s = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(B, C, 1, 1)
        return x * s


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        m = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * m


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cg = ChannelGate(c)
        self.sg = SpatialGate()

    def forward(self, x):
        return self.sg(self.cg(x))


class CBAMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(dim),
        )
        self.cbam = CBAM(dim)

    def forward(self, x):
        h = self.body(x)
        h = self.cbam(h)
        return x + h


class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        c = in_channels
        c_ = max(1, c // reduction)

        self.theta = spectral_norm(nn.Conv2d(c, c_, 1, bias=False))
        self.phi = spectral_norm(nn.Conv2d(c, c_, 1, bias=False))
        self.g = spectral_norm(nn.Conv2d(c, c_, 1, bias=False))
        self.out = spectral_norm(nn.Conv2d(c_, c, 1, bias=False))

        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.scale = c_ ** 0.5

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.theta(x).view(B, -1, N)
        k = self.phi(x).view(B, -1, N)
        v = self.g(x).view(B, -1, N)
        scale = q.size(1) ** 0.5
        energy = torch.bmm(q.transpose(1, 2), k) / (scale * 2.0)  # 스케일 보정
        attn = torch.softmax(energy, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        y = torch.bmm(v, attn.transpose(1, 2)).view(B, -1, H, W)
        y = self.out(y)
        return x + self.gamma * y  # residual


class AttnBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn = SelfAttention2d(dim)

    def forward(self, x):
        h = self.pre(x)
        h = self.attn(h)
        return x + h


class AttnGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, ngf=64,
                 n_blocks=4, attn_kind="sagan", n_attn=4):
        super().__init__()

        dim1 = ngf  # 128
        dim2 = ngf * 2  # 256
        dim3 = ngf * 4  # 512

        # ----------------
        # Encoder (feature taps)
        # ----------------
        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels, dim1, 7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(dim1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(dim1, dim2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(dim2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(dim2, dim3, 3, stride=2, padding=1),
            nn.InstanceNorm2d(dim3),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ----------------
        # Body: residual blocks (n_res) + attention blocks
        # ----------------
        dim = dim3
        Attn = CBAMBlock
        self.attn_blocks = nn.ModuleList([Attn(dim) for _ in range(n_attn)])

        # ----------------
        # Decoder with skip-concat
        # ----------------
        # up1: dim3 -> dim2, then concat enc1(dim2) => dim2+dim2
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim3, dim2, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(dim2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(dim2 + dim2, dim2, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(dim2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # up2: dim2 -> dim1, then concat enc0(dim1) => dim1+dim1
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(dim2, dim1, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(dim1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(dim1 + dim1, dim1, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(dim1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # ----------------
        # Final: concat with input image (like you already do)
        # ----------------
        self.final = nn.Sequential(
            nn.Conv2d(dim1, dim1 // 2, 3, padding=1, padding_mode='replicate'),
            nn.InstanceNorm2d(dim1 // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim1 // 2, out_channels, 3, padding=1, padding_mode='replicate'),
            # nn.Tanh()
        )
        nn.init.constant_(self.final[-1].weight, 0.0001)
        nn.init.constant_(self.final[-1].bias, 0.0)

    def forward(self, x):
        img = x  # for final concat

        # Encoder + save skips
        e0 = self.enc0(x)  # (B, dim1, H,   W)
        e1 = self.enc1(e0)  # (B, dim2, H/2, W/2)
        x = self.enc2(e1)  # (B, dim3, H/4, W/4)
        # Attention
        for attn in self.attn_blocks:
            x = attn(x)

        # Decoder + skip concat
        x = self.up1(x)  # (B, dim2, H/2, W/2)
        x = torch.cat([x, e1], dim=1)  # (B, dim2+dim2, H/2, W/2)
        x = self.fuse1(x)  # (B, dim2, ...)

        x = self.up2(x)  # (B, dim1, H, W)
        x = torch.cat([x, e0], dim=1)  # (B, dim1+dim1, H, W)
        x = self.fuse2(x)  # (B, dim1, H, W)

        # Final: concat with original input (optional but you already had it)
        # x = torch.cat([x, img], dim=1)  # (B, dim1+in_channels, H, W)
        x = self.final(x)

        return x


class ResidualBlock(nn.Module):
    """표준적인 ResNet 블록에 Attention을 내장"""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )
        self.attn = CBAMBlock(dim)  # 기존 사용하시던 CBAM 유지

    def forward(self, x):
        return x + self.attn(self.block(x))


class ModernAttnGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, ngf=64, n_blocks=6):
        super().__init__()

        # [Encoder] 깊이감을 주되, Spectral Norm으로 안정성 확보
        self.begin = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        # Downsampling (Strided Conv)
        self.down1 = self._make_down_block(ngf, ngf * 2)  # 1/2
        self.down2 = self._make_down_block(ngf * 2, ngf * 4)  # 1/4

        # [Bottleneck] Residual + Attention Blocks
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks += [ResidualBlock(ngf * 4)]
        self.bottleneck = nn.Sequential(*res_blocks)

        # [Decoder] PixelShuffle을 이용한 정교한 업샘플링
        self.up1 = self._make_up_block(ngf * 4, ngf * 2)  # 2/4 -> 1/2
        self.up2 = self._make_up_block(ngf * 2, ngf)  # 1/2 -> 1

        # [Final Layer]
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, 7),
            nn.Tanh()  # CycleGAN은 보통 -1 ~ 1 범위를 위해 Tanh 사용
        )

    def _make_down_block(self, in_f, out_f):
        return nn.Sequential(
            nn.Conv2d(in_f, out_f, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_f),
            nn.ReLU(True)
        )

    def _make_up_block(self, in_f, out_f):
        # PixelShuffle은 채널을 4배로 늘려 해상도를 2배 키움
        return nn.Sequential(
            nn.Conv2d(in_f, out_f * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(out_f),
            nn.ReLU(True)
        )

    def forward(self, x):
        # Skip Connection을 위한 리스트
        d0 = self.begin(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)

        b = self.bottleneck(d2)

        # U-Net Skip Connection 적용 (Concat)
        u1 = self.up1(b)
        u2 = self.up2(u1 + d1)  # 단순 더하기 혹은 torch.cat 가능

        return self.final(u2 + d0)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class AttnDiscriminator(nn.Module):
    def __init__(self, in_channels=1, ndf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1)  # 32x32
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1)  # 16x16
        self.norm2 = nn.InstanceNorm2d(ndf * 2)

        # Attention block at 16x16 stage
        self.cbam = CBAM(ndf * 2)

        self.conv3 = nn.Conv2d(ndf * 2, 1, 4, stride=1, padding=1)  # Patch score
        # output ~ [B,1,15,15]

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.norm2(self.conv2(x)))
        x = self.cbam(x)
        x = self.conv3(x)
        return x


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=7, stride=1, padding='same'),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=7, stride=1, padding='same', bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=3, padding='same', stride=1, bias=use_bias),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
