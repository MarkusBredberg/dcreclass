import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm


def load_gan_generator(path, latent_dim, device='cpu'):
    generator = DCGANGenerator(latent_dim=latent_dim).to(device)
    state_dict = torch.load(path, map_location=device)
    model_state = generator.state_dict()
    filtered_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state_dict[k] = v
            else:
                print(f"Skipping key '{k}': checkpoint shape {v.shape} does not match model shape {model_state[k].shape}.")
        else:
            print(f"Skipping unexpected key '{k}' in checkpoint.")
    # Update the model state with the filtered checkpoint values.
    model_state.update(filtered_state_dict)
    generator.load_state_dict(model_state)
    return generator


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=128, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            # Input: (batch_size, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),   # 4x4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),        # 8x8
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),        # 16x16
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),              # 32x32
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),             # 64x64
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),              # 128x128
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, x):
        return self.main(x)


class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=1):
        """
        Args:
            ndf (int): Base number of discriminator feature maps.
            nc (int): Number of channels in input images.
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Changed kernel size from 5 to 4 to match input dimensions.
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            # No sigmoid because we use raw logits with custom loss functions
        )
    
    def forward(self, x):
        out = self.main(x)
        # Average over spatial dimensions
        out = torch.mean(out, dim=[2, 3], keepdim=True)
        return out.view(-1, 1)
    
    
##########################################
########## ADVANCED STUFF ################
##########################################
class DualInputDiscriminator(nn.Module):
    def __init__(self, img_shape, scat_shape, output_dim=1, hidden_dim=128, disc_hidden_dim=384, dropout_rate=0.3, J=2):
        super().__init__()
        # --- Image Branch ---
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_to_latent_img = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # --- Scattering Coefficients Branch ---
        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        if J == 4:
            downsample_blocks = 1
        elif J == 3:
            downsample_blocks = 2
        elif J == 2:
            downsample_blocks = 3
        elif J == 1:
            downsample_blocks = 4
        else:
            raise ValueError("Invalid value for J. Supported values are 1, 2, 3, or 4.")
        conv_blocks = []
        for _ in range(downsample_blocks):
            conv_blocks.append(conv_block(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1))
            conv_blocks.append(conv_block(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1))
        self.conv_to_latent_scat = nn.Sequential(
            conv_block(scat_shape[0], hidden_dim, kernel_size=3, stride=1, padding=1),
            *conv_blocks
        )

        # --- Fully Connected Layers ---
        # Use dummy inputs to determine flattened dimensions.
        with torch.no_grad():
            dummy_img = torch.zeros(1, *img_shape)
            dummy_scat = torch.zeros(1, *scat_shape)
            # Process image branch
            img_feat = self.cnn_encoder(dummy_img)
            img_feat = self.conv_to_latent_img(img_feat)
            img_flat_dim = img_feat.view(1, -1).size(1)
            # Process scattering branch and pool to a fixed size (e.g. 4x4)
            scat_feat = self.conv_to_latent_scat(dummy_scat)
            scat_feat = F.adaptive_avg_pool2d(scat_feat, output_size=(4, 4))
            scat_flat_dim = scat_feat.view(1, -1).size(1)
            combined_dim = img_flat_dim + scat_flat_dim

        self.FC_input = nn.Linear(combined_dim, disc_hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(disc_hidden_dim)
        self.FC_hidden = nn.Linear(disc_hidden_dim, disc_hidden_dim // 2, bias=True)
        self.bn2 = nn.BatchNorm1d(disc_hidden_dim // 2)
        self.FC_out = nn.Linear(disc_hidden_dim // 2, output_dim, bias=True)
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, img, scat):
        # Process image branch.
        img_features = self.cnn_encoder(img)
        img_features = self.conv_to_latent_img(img_features)
        img_flat = img_features.view(img_features.size(0), -1)
        
        # Process scattering branch.
        scat_features = self.conv_to_latent_scat(scat)
        # Force the scattering features to a fixed spatial size (4x4)
        scat_features = F.adaptive_avg_pool2d(scat_features, output_size=(4, 4))
        scat_flat = scat_features.view(scat_features.size(0), -1)
        
        # Concatenate the flattened features.
        combined = torch.cat([img_flat, scat_flat], dim=1)
        h = self.FC_input(combined)
        h = self.LeakyReLU(self.bn1(h))
        h = self.dropout(h)
        h = self.FC_hidden(h)
        h = self.LeakyReLU(self.bn2(h))
        h = self.dropout(h)
        out = self.FC_out(h)
        return out





# ------------------------------------------------
# Optional Self-Attention block
# ------------------------------------------------
class SelfAttention(nn.Module):
    """
    Simple Self-Attention block from SAGAN (Zhang et al.).
    """
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, w, h = x.size()
        # Compute queries and keys
        proj_query = self.query_conv(x).view(b, -1, w*h)    # (B, C//8, N)
        proj_key   = self.key_conv(x).view(b, -1, w*h)        # (B, C//8, N)
        energy     = torch.bmm(proj_query.transpose(1,2), proj_key)  # (B, N, N)
        attention  = self.softmax(energy)

        # Compute values
        proj_value = self.value_conv(x).view(b, -1, w*h)      # (B, C, N)
        out = torch.bmm(proj_value, attention.transpose(1,2)) # (B, C, N)
        out = out.view(b, c, w, h)
        # Learnable scaling
        out = self.gamma * out + x
        return out


# ------------------------------------------------
# Improved Generator
# ------------------------------------------------
class GANGenerator(nn.Module):
    """
    An improved DCGAN-like generator with optional self-attention.
    """
    def __init__(self, latent_dim=100, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            # Input: (N, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # Insert self-attention at 32×32 (after upsampling from 16×16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            SelfAttention(ngf * 2),  # Self-attention at 32×32 resolution

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )

        # Optional: initialize weights
        self.apply(self._weights_init)

    def forward(self, x):
        return self.main(x)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)


# ------------------------------------------------
# Improved Discriminator
# ------------------------------------------------
class GANDiscriminator(nn.Module):
    """
    An improved DCGAN-like discriminator with spectral normalization
    and an optional self-attention block.
    """
    def __init__(self, ndf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            # Input: (N, nc, 128, 128)
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # Insert self-attention around 32×32 resolution
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(ndf * 4),

            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            # Final 4×4 -> 1×1
            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
        )

        # Optional: initialize weights
        self.apply(self._weights_init)

    def forward(self, x):
        out = self.main(x)
        return out.view(-1, 1)  # raw logits

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        # We usually do NOT apply batchnorm in the improved discriminator,
        # but if you do, you can init it similarly.



class ResBlockUp(nn.Module):
    """
    Residual block that upsamples the feature map, then applies convolutions.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')  # or 'bilinear'
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1x1 convolution to match dimensions in skip if channels differ
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        
        self.relu = nn.ReLU(True)

    def forward(self, x):
        # Upsample first
        out = self.upsample(x)
        
        # First conv
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip branch (also upsample, then match channels if needed)
        skip = self.upsample(x)
        skip = self.skip_conv(skip)
        
        # Combine skip connection
        out = out + skip
        out = self.relu(out)
        
        return out
    
class ResNetGenerator(nn.Module):
    def __init__(self, latent_dim=128, ngf=64, nc=1):
        super().__init__()
        self.linear = nn.Linear(latent_dim, ngf * 8 * 4 * 4)  # Start with a 4x4 feature map
        
        self.block1 = ResBlockUp(ngf * 8, ngf * 4)  # 4x4  -> 8x8
        self.block2 = ResBlockUp(ngf * 4, ngf * 2)  # 8x8  -> 16x16
        self.block3 = ResBlockUp(ngf * 2, ngf)      # 16x16 -> 32x32
        self.block4 = ResBlockUp(ngf, ngf // 2)     # 32x32 -> 64x64
        self.block5 = ResBlockUp(ngf // 2, ngf // 4)  # 64x64 -> 128x128

        self.conv_final = nn.Conv2d(ngf // 4, nc, 3, 1, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, z):
        # Project and reshape latent vector
        out = self.linear(z).view(z.size(0), -1, 4, 4)
        
        # Pass through residual up blocks
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        
        # Final conv + Tanh to get output in [-1,1]
        out = self.conv_final(out)
        return self.tanh(out)
    
class MLPGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=512):
        super(MLPGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z).view(z.size(0), -1)

class SelfAttention(nn.Module):
    """
    Simple Self-Attention block from SAGAN (Zhang et al.).
    """
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv   = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, w, h = x.size()
        query = self.query_conv(x).view(b, -1, w*h)   # (B, C//8, N)
        key   = self.key_conv(x).view(b, -1, w*h)     # (B, C//8, N)
        energy = torch.bmm(query.transpose(1, 2), key)  # (B, N, N)
        attention = self.softmax(energy)

        value = self.value_conv(x).view(b, -1, w*h)   # (B, C, N)
        out   = torch.bmm(value, attention.transpose(1, 2))  # (B, C, N)
        out   = out.view(b, c, w, h)
        return self.gamma * out + x  # learnable scaling

class AdvancedGenerator(nn.Module):
    """
    A generator that:
      1) Uses a linear projection from the latent space into a 4×4 feature map.
      2) Upscales step by step with ConvTranspose2d to reach 128×128.
      3) Inserts a single Self-Attention block at ~16×16.
    """
    def __init__(self, latent_dim=100, ngf=64, nc=1):
        super().__init__()
        # 1. Project the latent vector into a 4×4 feature map with (ngf*8) channels
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, ngf * 8 * 4 * 4),
            nn.ReLU(True),
        )

        # 2. Define the upsampling blocks
        self.conv_blocks = nn.Sequential(
            # 4×4 → 8×8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # 8×8 → 16×16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # Insert a single Self-Attention block at ~16×16
            SelfAttention(ngf * 2),

            # 16×16 → 32×32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # 32×32 → 64×64
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            # 64×64 → 128×128
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(self._weights_init)

    def forward(self, z):
        # 1) Map latent vector (z) to 4×4×(ngf*8)
        out = self.linear(z.view(z.size(0), -1))
        out = out.view(z.size(0), -1, 4, 4)

        # 2) Pass through the upsampling + self-attention blocks
        out = self.conv_blocks(out)
        return out

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
