import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super(CNN, self).__init__()
        
        in_channels = input_shape[0]
        
        self.features = nn.Sequential(
            # Block 1: 128→64, 16 channels
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2), 
            nn.Dropout2d(0.15),  # Reduced from 0.2
            
            # Block 2: 64→32, 32 channels
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.20),  # Progressive increase
            
            # Block 3: 32→16, 32 channels (keep same size - refinement)
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.25),
            
            # Block 4: 16→8, 48 channels
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.30),
            
            # Block 5: 8→4, 48 channels
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.35),
            
            # Block 6: 4→2, 64 channels (extra spatial reduction)
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.40),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier with extra regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 32), 
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    

class ScatterNet(nn.Module):
    """
    CNN-based classifier for scattering coefficients.
    Processes spatial structure in scattering features like TinyScatterSqueezeNet's scattering branch.
    """
    def __init__(self, input_dim, num_classes=2, hidden_dim=16, dropout_rate=0.5, J=2):
        """
        Args:
            input_dim: Not used (kept for compatibility)
            num_classes: Number of output classes
            hidden_dim: Number of channels in conv layers
            dropout_rate: Dropout probability
            J: Scattering scale parameter (determines downsampling)
        """
        super(ScatterNet, self).__init__()
        
        # Scattering input is [B, 169, 32, 32]
        C_scat = 169
        
        # Helper for conv blocks with dropout
        def conv_block(in_ch, out_ch, k, s, p, dropout_p=0.2):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout_p)
            )
        
        # Determine downsampling stages based on J (same as TinyScatterSqueezeNet)
        if J == 4:
            downsample_blocks = 1
        elif J == 3:
            downsample_blocks = 2
        elif J == 2:
            downsample_blocks = 3
        elif J == 1:
            downsample_blocks = 4
        else:
            downsample_blocks = 3  # Default for J=2
        
        # Build scattering processing layers
        scat_blocks = []
        
        # Initial conv to reduce channels: 169 → hidden_dim
        scat_blocks.append(conv_block(C_scat, hidden_dim, 3, 1, 1, dropout_p=0.2))
        scat_blocks.append(SEBlock(hidden_dim))
        
        # Repeated downsampling with increasing dropout
        for i in range(downsample_blocks):
            dropout_p = 0.2 + i * 0.1
            scat_blocks.append(conv_block(hidden_dim, hidden_dim, 3, 1, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim))
            scat_blocks.append(conv_block(hidden_dim, hidden_dim, 3, 2, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim))
        
        self.scat_encoder = nn.Sequential(*scat_blocks)
        
        # Compute output size after conv layers
        with torch.no_grad():
            dummy_scat = torch.zeros(1, C_scat, 32, 32)
            scat_f = self.scat_encoder(dummy_scat)
            feature_dim = scat_f.view(1, -1).size(1)
        
        # Classifier head (matching TinyScatterSqueezeNet style)
        self.fc1 = nn.Linear(feature_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        # Expected input: [B, 169, 32, 32]
        
        # Process scattering coefficients with spatial structure preserved
        x = self.scat_encoder(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classifier head
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.act(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))
    

#DualCSN
class DualCNNSqueezeNet(nn.Module):
    """
    Dual-branch CNN with complementary feature extraction.
    """
    def __init__(self, input_shape, num_classes=2, hidden_dim1=32, classifier_hidden_dim=32):
        super(DualCNNSqueezeNet, self).__init__()
        
        in_channels, H, W = input_shape
        
        # ==========================================
        # Branch 1: Image Encoder (large receptive field)
        # ==========================================
        self.img_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(8, momentum=0.1), 
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3), 
            
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3), 
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4)
        )
        
        # Compression with stronger regularization
        self.img_to_latent = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4),  
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4),  
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5)
        )
        
        # ==========================================
        # Branch 2: Detail Encoder (small receptive field with SE attention)
        # ==========================================
        def conv_block(in_ch, out_ch, k, s, p, dropout_p=0.3):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=True),
                nn.BatchNorm2d(out_ch, momentum=0.1),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout_p)
            )
        
        hidden_dim2 = 32
        detail_blocks = []
        
        # Initial conv + SE
        detail_blocks.append(conv_block(in_channels, hidden_dim2, 3, 1, 1, dropout_p=0.3))
        detail_blocks.append(SEBlock(hidden_dim2))
        
        # Three downsampling stages with increased dropout
        for i in range(3):
            dropout_p = 0.3 + i * 0.1
            # Non-strided conv + SE
            detail_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 1, 1, dropout_p=dropout_p))
            detail_blocks.append(SEBlock(hidden_dim2))
            # Strided conv + SE (downsampling)
            detail_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 2, 1, dropout_p=dropout_p))
            detail_blocks.append(SEBlock(hidden_dim2))
        
        self.detail_to_latent = nn.Sequential(*detail_blocks)
        
        # Compute combined feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, H, W)
            
            img_f = self.img_encoder(dummy_input)
            img_f = self.img_to_latent(img_f)
            img_dim = img_f.view(1, -1).size(1)
            
            detail_f = self.detail_to_latent(dummy_input)
            detail_dim = detail_f.view(1, -1).size(1)
            
            combined_dim = img_dim + detail_dim
        
        # ==========================================
        # Classifier Head
        # ==========================================
        self.FC_input = nn.Linear(combined_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1, momentum=0.1)  
        self.dropout1 = nn.Dropout(0.5)  
        
        self.FC_hidden = nn.Linear(hidden_dim1, classifier_hidden_dim)
        self.bn2 = nn.BatchNorm1d(classifier_hidden_dim, momentum=0.1)  
        self.dropout2 = nn.Dropout(0.5)  
        
        self.FC_classifier = nn.Linear(classifier_hidden_dim, num_classes)
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        # Branch 1: Image encoding path (large receptive field)
        x_img = self.img_encoder(x)
        x_img = self.img_to_latent(x_img)
        
        # Branch 2: Detail encoding path (small receptive field + attention)
        x_detail = self.detail_to_latent(x)
        
        # Flatten both branches
        x_img = x_img.view(x_img.size(0), -1)
        x_detail = x_detail.view(x_detail.size(0), -1)
        
        # Concatenate features
        x = torch.cat([x_img, x_detail], dim=1)
        
        # MLP classifier head
        x = self.act(self.bn1(self.FC_input(x)))
        x = self.dropout1(x)
        x = self.act(self.bn2(self.FC_hidden(x)))
        x = self.dropout2(x)
        
        return self.FC_classifier(x)


class DualScatterSqueezeNet(nn.Module): # DualSSN
    def __init__(self, img_shape, scat_shape, num_classes,
                 hidden_dim1=32, hidden_dim2=16, classifier_hidden_dim=32,
                 dropout_rate=0.5, J=2): 
        super(DualScatterSqueezeNet, self).__init__()
        C_img, H_img, W_img = img_shape
        C_scat, H_scat, W_scat = scat_shape

        # Image Branch with Dropout
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(C_img, 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),  
            
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),  
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)  
        )
        
        self.conv_to_latent_img = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),  
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),  
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),  
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4)  
        )

        # Scattering branch conv_block helper
        def conv_block(in_ch, out_ch, k, s, p, dropout_p=0.2): 
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout_p) 
            )

        # Determine number of downsampling stages
        if J == 4:
            downsample_blocks = 1
        elif J == 3:
            downsample_blocks = 2
        elif J == 2:
            downsample_blocks = 3
        elif J == 1:
            downsample_blocks = 4
        else:
            raise ValueError("J must be 1, 2, 3, or 4")

        # Build scattering branch with SE blocks and dropout
        scat_blocks = []
        # initial conv + SE
        scat_blocks.append(conv_block(C_scat, hidden_dim2, 3, 1, 1, dropout_p=0.2))
        scat_blocks.append(SEBlock(hidden_dim2))
        # repeated downsampling + SE
        for i in range(downsample_blocks):
            dropout_p = 0.2 + i * 0.1 
            scat_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 1, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim2))
            scat_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 2, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim2))

        self.conv_to_latent_scat = nn.Sequential(*scat_blocks)

        # Compute combined feature size
        with torch.no_grad():
            dummy_img  = torch.zeros(1, C_img, H_img, W_img)
            dummy_scat = torch.zeros(1, C_scat, H_scat, W_scat)
            img_f = self.cnn_encoder(dummy_img)
            img_f = self.conv_to_latent_img(img_f)
            scat_f = self.conv_to_latent_scat(dummy_scat)
            img_dim  = img_f.view(1, -1).size(1)
            scat_dim = scat_f.view(1, -1).size(1)
            combined_dim = img_dim + scat_dim

        # Classifier Head
        self.FC_input       = nn.Linear(combined_dim, hidden_dim1)
        self.bn1            = nn.BatchNorm1d(hidden_dim1)
        self.dropout1       = nn.Dropout(0.5)  
        self.FC_hidden      = nn.Linear(hidden_dim1, classifier_hidden_dim)
        self.bn2            = nn.BatchNorm1d(classifier_hidden_dim)
        self.dropout2       = nn.Dropout(0.5)  
        self.FC_classifier  = nn.Linear(classifier_hidden_dim, num_classes)
        self.act            = nn.LeakyReLU(0.2)

    def forward(self, img, scat):
        # Image path
        x_img = self.cnn_encoder(img)
        x_img = self.conv_to_latent_img(x_img)
        # Scattering path
        x_scat = self.conv_to_latent_scat(scat)
        # Flatten and concat
        x_img = x_img.view(x_img.size(0), -1)
        x_scat = x_scat.view(x_scat.size(0), -1)
        x = torch.cat([x_img, x_scat], dim=1)
        # MLP head 
        x = self.act(self.bn1(self.FC_input(x)))
        x = self.dropout1(x)  
        x = self.act(self.bn2(self.FC_hidden(x)))
        x = self.dropout2(x)  
        return self.FC_classifier(x)