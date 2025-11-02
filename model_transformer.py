import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# base class for other attention modules to inherit from
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

class UTransform(nn.Module):
    """
    https://github.com/HXLH50K/U-Net-Transformer/blob/main/models/utransformer/U_Transformer.py
    https://arxiv.org/pdf/2103.06104.pdf
    """

    def __init__(self, in_channels=4, classes=3, scale_factor=2):
        super(UTransform, self).__init__()

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.MHSA = MultiHeadSelfAttention(512)
        self.up1 = TransformerUp(512, 256)
        self.up2 = TransformerUp(256, 128)
        self.scale_factor = scale_factor

        if scale_factor == 2:
            self.up3 = TransformerUp(128, 64)
            # The output of up3 is 64 channels, so ConvTranspose2d should start with 64
            self.up_sample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2) 
            last = 64
        else:  # scale_factor == 4
            # Assuming up3 output is 128 (from TransformerUp(256, 128))
            self.up3 = TransformerUp(256, 128) 
            self.up_sample = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
            self.up_sample2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
            last = 128

        self.outc = OutConv(last, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.MHSA(x4)
        
        # The 'S' (skip) input for TransformerUp should be the corresponding layer
        # from the encoder path.
        x = self.up1(x4, x3) # x4 (from MHSA) and x3 (skip)
        x = self.up2(x, x2)  # x (from up1) and x2 (skip)
        x = self.up3(x, x1)  # x (from up2) and x1 (skip)
        
        x = self.up_sample(x)
        if self.scale_factor > 2: 
            x = self.up_sample2(x)
        return self.outc(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            *[
                nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1, bias=False), 
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1, bias=False), 
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True), 
            ]
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            *[
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels),
            ]
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class TransformerUp(nn.Module):
    def __init__(self, Ychannels, Schannels):
        super(TransformerUp, self).__init__()
        self.MHCA = MultiHeadCrossAttention(Ychannels, Schannels)
        
        self.conv = nn.Sequential(
            nn.Conv2d(
                Ychannels, Schannels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                Schannels, Schannels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(Schannels),
            nn.ReLU(inplace=True),
        )

    def forward(self, Y, S):
        x = self.MHCA(Y, S)
        x = self.conv(x)
        return x



class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor one wants to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        # --- Ensures channels is at least 1, and handle odd numbers
        if channels % 2 != 0:
            channels += 1 
        half_channels = channels // 2
        self.channels = half_channels 
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_channels, 2).float() / half_channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        
        # Use torch.arange with correct device and dtype
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        
        # Handle the case where self.channels (half_channels) was based on an odd orig_ch
        # The original code was dividing by 2 and ceiling, which is complex.
        # A simpler way is to just create the full embedding and slice it.
        
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)

        # Expand dims to broadcast
        emb_x = emb_x.unsqueeze(1).repeat(1, y, 1) # (x, y, half_channels)
        emb_y = emb_y.unsqueeze(0).repeat(x, 1, 1) # (x, y, half_channels)
        
        emb = torch.cat((emb_x, emb_y), dim=-1) # (x, y, channels)

        # Slice to match original channel dimension if it was odd
        emb = emb[:, :, :orig_ch]
        
        return emb.unsqueeze(0).repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        # (B, C, H, W) -> (B, H, W, C)
        tensor_permuted = tensor.permute(0, 2, 3, 1) 
        enc = self.penc(tensor_permuted)
        # (B, H, W, C) -> (B, C, H, W)
        return enc.permute(0, 3, 1, 2) 


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = nn.Linear(channel, channel, bias=False)
        self.key = nn.Linear(channel, channel, bias=False)
        self.value = nn.Linear(channel, channel, bias=False)
        self.pe = PositionalEncodingPermute2D(channel)

    def forward(self, x):
        b, c, h, w = x.size()
        pe = self.pe(x)
        x_with_pe = x + pe
        # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x_flat = x_with_pe.flatten(2).transpose(1, 2)  
        Q = self.query(x_flat)
        K = self.key(x_flat)
        V = self.value(x_flat)
        x_attn = F.scaled_dot_product_attention(Q, K, V) 
        # (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        x_out = x_attn.transpose(1, 2).reshape(b, c, h, w)
        return x + x_out


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS):
        super(MultiHeadCrossAttention, self).__init__()
        
        self.Sconv = nn.Sequential(
            nn.MaxPool2d(2), # -> (B, 256, H/8, W/8)
            nn.Conv2d(channelS, channelS, kernel_size=1, bias=False), # -> (B, 256, H/8, W/8)
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True),
        )
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1, bias=False), # -> (B, 256, H/8, W/8)
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True),
        )
        
        self.query = nn.Linear(channelS, channelS, bias=False) # Query from Y
        self.key = nn.Linear(channelS, channelS, bias=False)   # Key from S
        self.value = nn.Linear(channelS, channelS, bias=False) # Value from S
        
        self.conv = nn.Sequential(
            # Input is 'channelS' from attention output
            nn.Conv2d(channelS, channelS, kernel_size=1, bias=False), 
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), # -> (B, 256, H/4, W/4)
        )
        
        # This projects Y up to S's spatial dims and concatenates
        # Y = (B, 512, H/8, W/8)
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), # -> (B, 512, H/4, W/4)
            nn.Conv2d(channelY, channelS, kernel_size=1, bias=False), # -> (B, 256, H/4, W/4)
            nn.BatchNorm2d(channelS),
            nn.ReLU(inplace=True),
        )
        
        # Positional encoding should be for the *projected* channel dim
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelS) # Y is projected to channelS

    def forward(self, Y, S):
        # Y = (B, Yc, Yh, Yw) e.g., (B, 512, H/8, W/8)
        # S = (B, Sc, Sh, Sw) e.g., (B, 256, H/4, W/4)
        
        # 1. Prepare Key and Value from Skip connection S
        S_down = self.Sconv(S) # -> (B, Sc, Yh, Yw) e.g., (B, 256, H/8, W/8)
        Spe = self.Spe(S_down)
        S_pe = S_down + Spe
        S_flat = S_pe.flatten(2).transpose(1, 2) # (B, Yh*Yw, Sc)
        
        K = self.key(S_flat)
        V = self.value(S_flat)

        # 2. Prepare Query from Decoder Y
        Y_proj = self.Yconv(Y) # -> (B, Sc, Yh, Yw) e.g., (B, 256, H/8, W/8)
        Ype = self.Ype(Y_proj)
        Y_pe = Y_proj + Ype
        Y_flat = Y_pe.flatten(2).transpose(1, 2) # (B, Yh*Yw, Sc)
        
        Q = self.query(Y_flat)

        # 3. Attention
        # Q, K, V are all (B, Yh*Yw, Sc)
        x_attn = F.scaled_dot_product_attention(Q, K, V) # (B, Yh*Yw, Sc)
        
        # 4. Reshape and process
        b, c, h, w = Y_proj.shape # Get shape from projected tensor
        x = x_attn.transpose(1, 2).reshape(b, c, h, w) # (B, Sc, Yh, Yw)
        x = x + Y_proj 
        Z = self.conv(x) # -> (B, Sc, Sh, Sw) e.g., (B, 256, H/4, W/4)
        Z = Z * S # Element-wise gate
        # 5. Prepare concatenated skip
        Y2 = self.Yconv2(Y) # -> (B, Sc, Sh, Sw) e.g., (B, 256, H/4, W/4)
        # Concatenate along channel dim
        return torch.cat([Z, Y2], dim=1) # (B, 2*Sc, Sh, Sw) e.g., (B, 512, H/4, W/4)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
