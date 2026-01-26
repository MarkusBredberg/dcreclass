#Works for J=2,3,4 Dual, MLP, CNN
import torch
import torch.nn as nn

NORM = ['NONE', 'BATCH', 'GROUP'][1]
if NORM=='NONE':
    def create_conv_block(in_channels, out_channels, kernel_size, stride, padding, use_pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.2),
        ]
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def create_fc_block(input_dim, output_dim, dropout=0.3):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )


    class MLP_Encoder(nn.Module):
        def __init__(self, input_channels, hidden_dim1=256, hidden_dim2=128, latent_dim=32, J=2):
            super(MLP_Encoder, self).__init__()

            # Determine architecture based on J
            if J == 4:
                downsample_blocks = 1
            elif J == 3:
                downsample_blocks = 2
            elif J == 2:
                downsample_blocks = 3
            else:
                raise ValueError("Invalid value for J. Supported values are [2, 3, 4].")
            
            conv_blocks = []
            for _ in range(downsample_blocks):
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=1, padding=1))
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=2, padding=1))

            self.conv_to_latent = nn.Sequential(
                create_conv_block(input_channels, hidden_dim2, kernel_size=3, stride=1, padding=1),
                *conv_blocks
            )
            
            # Fully connected layers
            self.FC_input = create_fc_block(hidden_dim2 * 4 * 4, hidden_dim1)
            self.FC_hidden = create_fc_block(hidden_dim1, hidden_dim2)
            self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
            self.FC_var = nn.Linear(hidden_dim2, latent_dim)

        def forward(self, x):
            h = self.conv_to_latent(x)
            h = h.view(h.size(0), -1)
            h = self.FC_input(h)
            h = self.FC_hidden(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar
        

    class Dual_Encoder(nn.Module):
        def __init__(self, scatshape, hidden_dim1=256, hidden_dim2=128, latent_dim=32, J=2):
            super(Dual_Encoder, self).__init__()
            
            # Define cnn_encoder for image features
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                nn.LeakyReLU(0.2)
            )

            # Determine architecture based on J
            if J == 4:
                downsample_blocks = 1
            elif J == 3:
                downsample_blocks = 2
            elif J == 2:
                downsample_blocks = 3
            else:
                raise ValueError("Invalid value for J. Supported values are [2, 3, 4].")
            
            # Define convolution blocks for scatter coefficients
            conv_blocks = []
            for _ in range(downsample_blocks):
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=1, padding=1))
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=2, padding=1))

            self.conv_to_latent_scat = nn.Sequential(
                create_conv_block(scatshape[-3], hidden_dim2, kernel_size=3, stride=1, padding=1),
                *conv_blocks
            )
            
            # Define conv_to_latent_img and other layers
            self.conv_to_latent_img = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 64x64 -> 32x32
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 32x32 -> 16x16
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 16x16 -> 8x8
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 8x8 -> 4x4
                nn.LeakyReLU(0.2)
            )

            # Fully connected layers
            self.FC_input = nn.Linear(128*2*4*4, 384, bias=True)
            self.FC_hidden = nn.Linear(384, 256, bias=True)
            self.FC_mean = nn.Linear(256, latent_dim, bias=True)
            self.FC_var = nn.Linear(256, latent_dim, bias=True)
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, img, scat):
            cnn_features = self.cnn_encoder(img)  # Use cnn_encoder for images
            img_h = self.conv_to_latent_img(cnn_features)
            scat_h = self.conv_to_latent_scat(scat)
            h = torch.cat((img_h, scat_h), dim=1)
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.FC_input(h))
            h = self.dropout(h)
            h = self.LeakyReLU(self.FC_hidden(h))
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar

        
    class CNN_Encoder(nn.Module):
        def __init__(self, latent_dim):
            super(CNN_Encoder, self).__init__()
            
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
                nn.LeakyReLU(0.2),

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2)
            )
            
            self.fc_layers = MLP_Encoder(input_channels=256, hidden_dim1=384, hidden_dim2=256, latent_dim=latent_dim, J=4)

        def forward(self, x):
            x = self.conv_layers(x)
            mean, logvar = self.fc_layers(x)
            return mean, logvar

    class CNN_Decoder(nn.Module):
        def __init__(self, latent_dim, intermediate_dim=256):
            super(CNN_Decoder, self).__init__()

            # Fully connected layers to transform latent vector into feature map
            self.fc_layers = nn.Sequential(
                nn.Linear(latent_dim, intermediate_dim * 4 * 4, bias=True),
                nn.LeakyReLU(0.2),
            )
            
            self.deconv_layers = nn.Sequential(
                # Upsample 4x4 -> 8x8, block of 3 layers
                nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 8x8 -> 16x16, block of 3 layers
                nn.Conv2d(intermediate_dim, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 16x16 -> 32x32, block of 3 layers
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 32x32 -> 64x64, block of 3 layers
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 64x64 -> 128x128, block of 3 layers
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Final layer to produce output with 1 channel (since input had 1 channel)
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid()  
            )

        def forward(self, z):
            z = self.fc_layers(z)
            z = z.view(z.size(0), -1, 4, 4)
            z = self.deconv_layers(z)
            return z

        
    class VAE_Model(nn.Module):
        def __init__(self, encoder, decoder):
            super(VAE_Model, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def reparameterize(self, mean, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std

        def forward(self, x):
            mean, logvar = self.encoder(x)
            z = self.reparameterize(mean, logvar)
            x_hat = self.decoder(z)
            return x_hat, mean, logvar
        
        def kl_divergence(self, x, mu, logvar):
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return kld / x.size(0)
        
    class Dual_VAE_Model(nn.Module):
        def __init__(self, encoder, decoder):
            super(Dual_VAE_Model, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def reparameterize(self, mean, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std

        def forward(self, img, scat):
            mean, logvar = self.encoder(img, scat)
            z = self.reparameterize(mean, logvar)
            x_hat = self.decoder(z)
            return x_hat, mean, logvar
        
        def kl_divergence(self, x, mu, logvar):
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return kld / x.size(0)


if NORM=='BATCH':
    def create_conv_block(in_channels, out_channels, kernel_size, stride, padding, use_pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        ]
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def create_fc_block(input_dim, output_dim, dropout=0.3):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )


    class MLP_Encoder(nn.Module):
        def __init__(self, input_channels, hidden_dim1=256, hidden_dim2=128, latent_dim=32, J=2):
            super(MLP_Encoder, self).__init__()

            # Determine architecture based on J
            if J == 4:
                downsample_blocks = 1
            elif J == 3:
                downsample_blocks = 2
            elif J == 2:
                downsample_blocks = 3
            else:
                raise ValueError("Invalid value for J. Supported values are [2, 3, 4].")
            
            conv_blocks = []
            for _ in range(downsample_blocks):
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=1, padding=1))
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=2, padding=1))

            self.conv_to_latent = nn.Sequential(
                create_conv_block(input_channels, hidden_dim2, kernel_size=3, stride=1, padding=1),
                *conv_blocks
            )
            
            # Fully connected layers
            self.FC_input = create_fc_block(hidden_dim2 * 4 * 4, hidden_dim1)
            self.FC_hidden = create_fc_block(hidden_dim1, hidden_dim2)
            self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
            self.FC_var = nn.Linear(hidden_dim2, latent_dim)

        def forward(self, x):
            h = self.conv_to_latent(x)
            h = h.view(h.size(0), -1)
            h = self.FC_input(h)
            h = self.FC_hidden(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar
        

    class Dual_Encoder(nn.Module):
        def __init__(self, scatshape, hidden_dim1=256, hidden_dim2=128, latent_dim=32, J=2):
            super(Dual_Encoder, self).__init__()
            
            # Define cnn_encoder for image features
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(32),  # Replaced GroupNorm with BatchNorm2d
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),  # Replaced GroupNorm with BatchNorm2d
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(128),  # Replaced GroupNorm with BatchNorm2d
                nn.LeakyReLU(0.2)
            )

            # Determine architecture based on J
            if J == 4:
                downsample_blocks = 1
            elif J == 3:
                downsample_blocks = 2
            elif J == 2:
                downsample_blocks = 3
            else:
                raise ValueError("Invalid value for J. Supported values are [2, 3, 4].")
            
            # Define convolution blocks for scatter coefficients
            conv_blocks = []
            for _ in range(downsample_blocks):
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=1, padding=1))
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=2, padding=1))

            self.conv_to_latent_scat = nn.Sequential(
                create_conv_block(scatshape[-3], hidden_dim2, kernel_size=3, stride=1, padding=1),
                *conv_blocks
            )
            
            # Define conv_to_latent_img and other layers
            self.conv_to_latent_img = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 64x64 -> 32x32
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 32x32 -> 16x16
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 16x16 -> 8x8
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 8x8 -> 4x4
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2)
            )

            # Fully connected layers
            self.FC_input = nn.Linear(128*2*4*4, 384, bias=True)
            self.bn1 = nn.BatchNorm1d(384)  # Replaced GroupNorm with BatchNorm1d
            self.FC_hidden = nn.Linear(384, 256, bias=True)
            self.bn2 = nn.BatchNorm1d(256)  # Replaced GroupNorm with BatchNorm1d
            self.FC_mean = nn.Linear(256, latent_dim, bias=True)
            self.FC_var = nn.Linear(256, latent_dim, bias=True)
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, img, scat):
            cnn_features = self.cnn_encoder(img)  # Use cnn_encoder for images
            img_h = self.conv_to_latent_img(cnn_features)
            scat_h = self.conv_to_latent_scat(scat)
            h = torch.cat((img_h, scat_h), dim=1)
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.bn1(self.FC_input(h)))  # Replaced GroupNorm with BatchNorm1d
            h = self.dropout(h)
            h = self.LeakyReLU(self.bn2(self.FC_hidden(h)))  # Replaced GroupNorm with BatchNorm1d
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar

        
    class CNN_Encoder(nn.Module):
        def __init__(self, latent_dim):
            super(CNN_Encoder, self).__init__()
            
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2)
            )
            
            self.fc_layers = MLP_Encoder(input_channels=256, hidden_dim1=384, hidden_dim2=256, latent_dim=latent_dim, J=4)

        def forward(self, x):
            x = self.conv_layers(x)
            mean, logvar = self.fc_layers(x)
            return mean, logvar

    class CNN_Decoder(nn.Module):
        def __init__(self, latent_dim, intermediate_dim=256):
            super(CNN_Decoder, self).__init__()

            # Fully connected layers to transform latent vector into feature map
            self.fc_layers = nn.Sequential(
                nn.Linear(latent_dim, intermediate_dim * 4 * 4, bias=True),
                nn.LeakyReLU(0.2),
            )
            
            self.deconv_layers = nn.Sequential(
                # Upsample 4x4 -> 8x8, block of 3 layers
                nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(intermediate_dim),
                nn.LeakyReLU(0.2),
                nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(intermediate_dim),
                nn.LeakyReLU(0.2),
                nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(intermediate_dim),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 8x8 -> 16x16, block of 3 layers
                nn.Conv2d(intermediate_dim, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 16x16 -> 32x32, block of 3 layers
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 32x32 -> 64x64, block of 3 layers
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 64x64 -> 128x128, block of 3 layers
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Final layer to produce output with 1 channel (since input had 1 channel)
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid()  
            )

        def forward(self, z):
            z = self.fc_layers(z)
            z = z.view(z.size(0), -1, 4, 4)
            z = self.deconv_layers(z)
            return z
        
    if True:
        class CNN_Decoder(nn.Module):
            def __init__(self, latent_dim, intermediate_dim=256):
                super(CNN_Decoder, self).__init__()

                # Fully connected layers to transform latent vector into feature map
                self.fc_layers = nn.Sequential(
                    nn.Linear(latent_dim, intermediate_dim * 4 * 4, bias=True),
                    nn.LeakyReLU(0.2),
                )
                
                self.deconv_layers = nn.Sequential(
                    # Upsample 4x4 -> 8x8, block of 3 layers
                    nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(intermediate_dim),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(intermediate_dim),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(intermediate_dim),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ConvTranspose2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),

                    # Upsample 8x8 -> 16x16, block of 3 layers
                    nn.Conv2d(intermediate_dim, 256, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),

                    # Upsample 16x16 -> 32x32, block of 3 layers
                    nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),

                    # Upsample 32x32 -> 64x64, block of 3 layers
                    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),

                    # Upsample 64x64 -> 128x128, block of 3 layers
                    nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(32),
                    nn.LeakyReLU(0.2),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),

                    # Final layer to produce output with 1 channel (since input had 1 channel)
                    nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.Sigmoid()  
                )

            def forward(self, z):
                z = self.fc_layers(z)
                z = z.view(z.size(0), -1, 4, 4)
                z = self.deconv_layers(z)
                return z

        
    class VAE_Model(nn.Module):
        def __init__(self, encoder, decoder):
            super(VAE_Model, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def reparameterize(self, mean, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std

        def forward(self, x):
            mean, logvar = self.encoder(x)
            z = self.reparameterize(mean, logvar)
            x_hat = self.decoder(z)
            return x_hat, mean, logvar
        
        def kl_divergence(self, x, mu, logvar):
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return kld / x.size(0)
        
    class Dual_VAE_Model(nn.Module):
        def __init__(self, encoder, decoder):
            super(Dual_VAE_Model, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def reparameterize(self, mean, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std

        def forward(self, img, scat):
            mean, logvar = self.encoder(img, scat)
            z = self.reparameterize(mean, logvar)
            x_hat = self.decoder(z)
            return x_hat, mean, logvar
        
        def kl_divergence(self, x, mu, logvar):
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return kld / x.size(0)
    
if NORM=='GROUP':
    def create_conv_block(in_channels, out_channels, kernel_size, stride, padding, use_pool=False, num_groups=32):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.GroupNorm(num_groups, out_channels),
            nn.LeakyReLU(0.2),
        ]
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def create_fc_block(input_dim, output_dim, dropout=0.3, num_groups=32):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.GroupNorm(num_groups, output_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )


    class MLP_Encoder(nn.Module):
        def __init__(self, input_channels, hidden_dim1=256, hidden_dim2=128, latent_dim=32, J=2, num_groups=32):
            super(MLP_Encoder, self).__init__()

            # Determine architecture based on J
            if J == 4:
                downsample_blocks = 1
            elif J == 3:
                downsample_blocks = 2
            elif J == 2:
                downsample_blocks = 3
            else:
                raise ValueError("Invalid value for J. Supported values are [2, 3, 4].")
            
            conv_blocks = []
            for _ in range(downsample_blocks):
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=1, padding=1, num_groups=num_groups))
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=2, padding=1, num_groups=num_groups))

            self.conv_to_latent = nn.Sequential(
                create_conv_block(input_channels, hidden_dim2, kernel_size=3, stride=1, padding=1, num_groups=num_groups),
                *conv_blocks
            )
            
            # Fully connected layers
            self.FC_input = create_fc_block(hidden_dim2 * 4 * 4, hidden_dim1, num_groups=num_groups)
            self.FC_hidden = create_fc_block(hidden_dim1, hidden_dim2, num_groups=num_groups)
            self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
            self.FC_var = nn.Linear(hidden_dim2, latent_dim)

        def forward(self, x):
            h = self.conv_to_latent(x)
            h = h.view(h.size(0), -1)
            h = self.FC_input(h)
            h = self.FC_hidden(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar
        

    class Dual_Encoder(nn.Module):
        def __init__(self, scatshape, hidden_dim1=256, hidden_dim2=128, latent_dim=32, J=2, num_groups=32):
            super(Dual_Encoder, self).__init__()
            
            # Define cnn_encoder for image features
            self.cnn_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 32),  
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 128),  
                nn.LeakyReLU(0.2)
            )

            # Determine architecture based on J
            if J == 4:
                downsample_blocks = 1
            elif J == 3:
                downsample_blocks = 2
            elif J == 2:
                downsample_blocks = 3
            else:
                raise ValueError("Invalid value for J. Supported values are [2, 3, 4].")
            
            # Define convolution blocks for scatter coefficients
            conv_blocks = []
            for _ in range(downsample_blocks):
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=1, padding=1, num_groups=num_groups))
                conv_blocks.append(create_conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=2, padding=1, num_groups=num_groups))

            self.conv_to_latent_scat = nn.Sequential(
                create_conv_block(scatshape[-3], hidden_dim2, kernel_size=3, stride=1, padding=1, num_groups=num_groups),
                *conv_blocks
            )
            
            # Define conv_to_latent_img and other layers
            self.conv_to_latent_img = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 64x64 -> 32x32
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 32x32 -> 16x16
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 16x16 -> 8x8
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),  # Downsample: 8x8 -> 4x4
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2)
            )

            # Fully connected layers
            self.FC_input = nn.Linear(128*2*4*4, 384, bias=True)
            self.bn1 = nn.GroupNorm(num_groups, 384)  
            self.FC_hidden = nn.Linear(384, 256, bias=True)
            self.bn2 = nn.GroupNorm(num_groups, 256)  
            self.FC_mean = nn.Linear(256, latent_dim, bias=True)
            self.FC_var = nn.Linear(256, latent_dim, bias=True)
            self.LeakyReLU = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(0.3)

        def forward(self, img, scat):
            cnn_features = self.cnn_encoder(img)  # Use cnn_encoder for images
            img_h = self.conv_to_latent_img(cnn_features)
            scat_h = self.conv_to_latent_scat(scat)
            h = torch.cat((img_h, scat_h), dim=1)
            h = h.view(h.size(0), -1)
            h = self.LeakyReLU(self.bn1(self.FC_input(h)))  
            h = self.dropout(h)
            h = self.LeakyReLU(self.bn2(self.FC_hidden(h)))  
            h = self.dropout(h)
            mean = self.FC_mean(h)
            logvar = self.FC_var(h)
            return mean, logvar

        
    class CNN_Encoder(nn.Module):
        def __init__(self, latent_dim, num_groups=32):
            super(CNN_Encoder, self).__init__()
            
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),

                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),

                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),

                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),

                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2)
            )
            
            self.fc_layers = MLP_Encoder(input_channels=256, hidden_dim1=384, hidden_dim2=256, latent_dim=latent_dim, J=4, num_groups=num_groups)

        def forward(self, x):
            x = self.conv_layers(x)
            mean, logvar = self.fc_layers(x)
            return mean, logvar

    class CNN_Decoder(nn.Module):
        def __init__(self, latent_dim, intermediate_dim=256, num_groups=32):
            super(CNN_Decoder, self).__init__()

            # Fully connected layers to transform latent vector into feature map
            self.fc_layers = nn.Sequential(
                nn.Linear(latent_dim, intermediate_dim * 4 * 4, bias=True),
                nn.LeakyReLU(0.2),
            )
            
            # Upsampling layers using Upsample and GroupNorm
            self.deconv_layers = nn.Sequential(
                # Upsample 4x4 -> 8x8, block of 3 layers
                nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, intermediate_dim),
                nn.LeakyReLU(0.2),
                nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, intermediate_dim),
                nn.LeakyReLU(0.2),
                nn.Conv2d(intermediate_dim, intermediate_dim, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, intermediate_dim),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 8x8 -> 16x16, block of 3 layers
                nn.Conv2d(intermediate_dim, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 256),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 16x16 -> 32x32, block of 3 layers
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 128),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 32x32 -> 64x64, block of 3 layers
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 64),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Upsample 64x64 -> 128x128, block of 3 layers
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(num_groups, 32),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode='nearest'),

                # Final layer to produce output with 1 channel (since input had 1 channel)
                nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Sigmoid()  # Output pixel values between 0 and 1
            )

        def forward(self, z):
            # Transform the latent vector into an intermediate feature map
            z = self.fc_layers(z)
            z = z.view(z.size(0), -1, 4, 4)
            # Apply the deconvolution and upsampling layers to reconstruct the image
            z = self.deconv_layers(z)
            return z

        
    class VAE_Model(nn.Module):
        def __init__(self, encoder, decoder):
            super(VAE_Model, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def reparameterize(self, mean, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std

        def forward(self, x):
            mean, logvar = self.encoder(x)
            z = self.reparameterize(mean, logvar)
            x_hat = self.decoder(z)
            return x_hat, mean, logvar
        
        def kl_divergence(self, x, mu, logvar):
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return kld / x.size(0)
        
    class Dual_VAE_Model(nn.Module):
        def __init__(self, encoder, decoder):
            super(Dual_VAE_Model, self).__init__()
            self.encoder = encoder
            self.decoder = decoder

        def reparameterize(self, mean, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std

        def forward(self, img, scat):
            mean, logvar = self.encoder(img, scat)
            z = self.reparameterize(mean, logvar)
            x_hat = self.decoder(z)
            return x_hat, mean, logvar
        
        def kl_divergence(self, x, mu, logvar):
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return kld / x.size(0)

        


###############################################
############ CONDITIONAL CNN VAE ##############
###############################################

class CMLP_Encoder(nn.Module):
    def __init__(self, input_channels, condition_dim, hidden_dim1=256, hidden_dim2=128, latent_dim=32):
        super(CMLP_Encoder, self).__init__()

        self.conv_to_latent = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim2, hidden_dim1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim1, hidden_dim1, kernel_size=2)
        )

        self.FC_input = nn.Linear(hidden_dim1 + condition_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)

        self.FC_hidden = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)

        self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
        self.FC_var = nn.Linear(hidden_dim2, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_dim2)

    def forward(self, x, condition):
        h = self.conv_to_latent(x)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, condition], dim=1)  # Concatenate condition with latent features
        h = self.LeakyReLU(self.bn1(self.FC_input(h)))
        h = self.dropout(h)
        h = self.LeakyReLU(self.layer_norm(self.FC_hidden(h)))
        h = self.dropout(h)
        mean = self.FC_mean(h)
        logvar = self.FC_var(h)
        return mean, logvar

class CCNN_Encoder(nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(CCNN_Encoder, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.conv_to_latent = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, latent_dim, kernel_size=2)
        )

        self.fc_layers = CMLP_Encoder(input_channels=256, hidden_dim1=384, hidden_dim2=256, latent_dim=latent_dim, condition_dim=condition_dim)

    def forward(self, x, condition):
        x = self.conv_layers(x)
        mean, logvar = self.fc_layers(x, condition)
        return mean, logvar

class CDual_Encoder(nn.Module):
    def __init__(self, scatshape, hidden_dim1, hidden_dim2, latent_dim, condition_dim):
        super(CDual_Encoder, self).__init__()
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_to_latent_img = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=2)
        )
        self.conv_to_latent_scat = nn.Sequential(
            nn.Conv2d(scatshape[-3], 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=2)
        )

        self.FC_input = nn.Linear(768 + condition_dim, hidden_dim2)
        self.bn1 = nn.BatchNorm1d(hidden_dim2)

        self.FC_hidden = nn.Linear(hidden_dim2, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)

        self.FC_mean = nn.Linear(hidden_dim2, latent_dim)
        self.FC_var = nn.Linear(hidden_dim2, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_dim2)
        #self.condition_fc = nn.Linear(128, 81)


    def forward(self, img, scat, condition):
        cnn_features = self.cnn_encoder(img)
        img_h = self.conv_to_latent_img(cnn_features)
        scat_h = self.conv_to_latent_scat(scat)
        h = torch.cat((img_h, scat_h), dim=1)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, condition], dim=1)
        h = self.LeakyReLU(self.bn1(self.FC_input(h)))
        h = self.dropout(h)
        h = self.LeakyReLU(self.layer_norm(self.FC_hidden(h)))
        h = self.dropout(h)
        mean = self.FC_mean(h)
        logvar = self.FC_var(h)
        return mean, logvar
    
class CCNN_Decoder(nn.Module):
    def __init__(self, latent_dim, condition_dim, intermediate_dim=None):
        if intermediate_dim is None:
            intermediate_dim = latent_dim + condition_dim
        super(CCNN_Decoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, intermediate_dim * 4 * 4),
            nn.ReLU(),
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(intermediate_dim, intermediate_dim // 2, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(intermediate_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(intermediate_dim // 2, 128, kernel_size=4, stride=2, padding=1),  # 16x16 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 128x128 
            nn.Sigmoid()
        )

    def forward(self, z, condition):
        z = torch.cat([z, condition], dim=1)
        z = self.fc_layers(z)
        z = z.view(z.size(0), -1, 4, 4)  # Reshape to the feature map shape (batch_size, channels, height, width)
        z = self.deconv_layers(z)
        return z


class CVAE_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(CVAE_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x, condition):
        mean, logvar = self.encoder(x, condition)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z, condition=condition)
        return x_hat, mean, logvar

    def kl_divergence(self, x, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / x.size(0)

class CDual_VAE_Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(CDual_VAE_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, img, scat, condition):
        mean, logvar = self.encoder(img, scat, condition)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decoder(z, condition)
        return x_hat, mean, logvar
    
    def kl_divergence(self, x, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kld / x.size(0)

    
def get_model(name, scatshape, hidden_dim1=None, hidden_dim2=None, latent_dim=None, num_classes=4, J=4):
    if 'MLP' in name and 'C' not in name:
        encoder = MLP_Encoder(input_channels=scatshape[-3], hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, latent_dim=latent_dim, J=J)
        decoder = CNN_Decoder(latent_dim=latent_dim, intermediate_dim=128)
        model = VAE_Model(encoder=encoder, decoder=decoder)
    elif 'MLP' in name:  # For conditional MLPs
        encoder = CMLP_Encoder(input_channels=scatshape[-3], hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, condition_dim=num_classes)
        decoder = CCNN_Decoder(latent_dim=latent_dim, condition_dim=num_classes) 
        model = CVAE_Model(encoder=encoder, decoder=decoder)
    elif name == "CNN":
        encoder = CNN_Encoder(latent_dim=latent_dim)
        decoder = CNN_Decoder(latent_dim=latent_dim, intermediate_dim=128)
        model = VAE_Model(encoder=encoder, decoder=decoder)
    elif name == 'DualCNN' or name == 'Dual':
        encoder = Dual_Encoder(scatshape=scatshape, latent_dim=latent_dim, J=J)
        decoder = CNN_Decoder(latent_dim=latent_dim, intermediate_dim=128)
        model = Dual_VAE_Model(encoder=encoder, decoder=decoder)
    elif name == "CCNN":
        encoder = CCNN_Encoder(latent_dim=latent_dim, condition_dim=num_classes)
        decoder = CCNN_Decoder(latent_dim=latent_dim, condition_dim=num_classes)
        model = CVAE_Model(encoder=encoder, decoder=decoder)
    elif name == "CDual":
        encoder = CDual_Encoder(scatshape=scatshape, hidden_dim1=512, hidden_dim2=256, latent_dim=latent_dim, condition_dim=num_classes)
        decoder = CCNN_Decoder(latent_dim=latent_dim, condition_dim=num_classes)
        model = CDual_VAE_Model(encoder=encoder, decoder=decoder)
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    return model

    
###############################################
################# CLASSIFIERS #################
###############################################

    
class CNN_Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Classifier, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # Assuming grayscale images as input
            nn.BatchNorm2d(32),  # Add batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),  # Add batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),  # Add batch normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.attention = nn.Sequential(
            nn.Linear(64 * 8 * 8, 64 * 8 * 8),  # Adjust the input size according to the output feature map size
            nn.Tanh(),
            nn.Linear(64 * 8 * 8, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*64, 64),  # Reduce the number of parameters by decreasing the number of units
            nn.BatchNorm1d(64),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Add dropout with 50% probability
            nn.Linear(64, num_classes)  # Ten outputs for ten classes
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        #attn_weights = self.attention(x)
        #x = x * attn_weights
        x = self.classifier(x)
        return x