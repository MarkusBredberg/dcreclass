import torch
import torch.nn as nn
from torch.autograd import Function

###############################################
####### Multilayer Perceptron #################
###############################################


# Scattering classifier
class MLPClassifier_legacy(nn.Module):
    def __init__(self, input_dim, num_classes=4, hidden_dim=120):
        super(MLPClassifier_legacy, self).__init__()
        # Compute second layer dimension such that when hidden_dim=120, fc2 becomes 84.
        fc2_dim = int(hidden_dim * 0.7)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input if necessary
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        #x = self.sigmoid(x)
        #return torch.softmax(x)
        return x # logits for BCE loss

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dim=128):
        """
        Lightweight MLP for scattering coefficients.
        Uses adaptive pooling to reduce spatial dimensions first.
        
        Args:
            input_dim: Expected flattened dimension (for validation) - NOT USED with adaptive pooling
            num_classes: Number of output classes
            hidden_dim: Hidden layer size (default 128)
        """
        super(MLPClassifier, self).__init__()
        
        # Adaptive pooling to reduce spatial dimensions
        # From [B, 169, 32, 32] -> [B, 169, 4, 4]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # New input dimension after pooling: 169 * 4 * 4 = 2704
        pooled_dim = 169 * 4 * 4
        
        # Smaller network appropriate for the task
        self.fc1 = nn.Linear(pooled_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Expected input: [B, C, H, W] e.g., [16, 169, 32, 32]
        
        # Apply adaptive pooling to reduce spatial dimensions
        # [B, 169, 32, 32] -> [B, 169, 4, 4]
        if x.dim() == 4:
            x = self.adaptive_pool(x)
        
        # Flatten: [B, 169, 4, 4] -> [B, 2704]
        x = x.view(x.size(0), -1)
        
        # Forward through network
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

  

################################################
#### Simple Convolutional Classifiers ##########
################################################

class RustigeClassifier(nn.Module):
    # From https://github.com/floriangriese/wGAN-supported-augmentation/blob/main/src/Classifiers/SimpleClassifiers/Classifiers.py
    def __init__(self, n_output_nodes=4):
        super(RustigeClassifier, self).__init__()

        self.conv_model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([64, 64]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([32, 32]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([16, 16]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LayerNorm([8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=2, stride=1, padding=0, bias=False)
        )

        self.fc_model = nn.Sequential(
            nn.Linear(16 * 7 * 7, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, n_output_nodes),
            nn.ReLU(True)
        )

    def forward(self, x):
        # image dimensions [128, 128]
        x = self.conv_model(x)
        # dimensions after convolution [7,7]

        # flatten for FC
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = self.fc_model(x)
        return x

class NEWRustigeClassifier(nn.Module):
    def __init__(self, input_shape, num_classes=4):
        super(NEWRustigeClassifier, self).__init__()
        
        # Parse (channels, height, width) from input_shape
        in_channels = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        # Convolutional block 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=8, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.ln1  = nn.LayerNorm([8, height // 2, width // 2])
        self.act1 = nn.LeakyReLU()
        
        # Convolutional block 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.ln2  = nn.LayerNorm([16, height // 4, width // 4])
        self.act2 = nn.LeakyReLU()
        
        # Convolutional block 3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.ln3  = nn.LayerNorm([32, height // 8, width // 8])
        self.act3 = nn.LeakyReLU()
        
        # Convolutional block 4
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.ln4  = nn.LayerNorm([16, height // 16, width // 16])
        self.act4 = nn.LeakyReLU()
        
        # Convolutional block 5
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.ln5  = nn.LayerNorm([16, height // 32, width // 32])
        self.act5 = nn.LeakyReLU()
        
        # Fully connected layers
        self.fc1   = nn.Linear(16 * (height // 32) * (width // 32), 100)
        self.act6  = nn.ReLU()
        self.fc2   = nn.Linear(100, num_classes)
        self.act7  = nn.ReLU()
        
        # Final softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass through conv blocks
        x = self.act1(self.ln1(self.conv1(x)))
        x = self.act2(self.ln2(self.conv2(x)))
        x = self.act3(self.ln3(self.conv3(x)))
        x = self.act4(self.ln4(self.conv4(x)))
        x = self.act5(self.ln5(self.conv5(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.act6(self.fc1(x))
        x = self.act7(self.fc2(x))
        
        # Final softmax
        x = self.softmax(x)
        return x

#Original
class BinaryClassifier_legacy(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super(BinaryClassifier_legacy, self).__init__()

        # Parse (channels, height, width) from input_shape
        in_channels = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        # Convolutional block 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=8, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(8)    
        self.act1 = nn.LeakyReLU()
        
        # Convolutional block 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.act2 = nn.LeakyReLU()
        
        # Convolutional block 3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.act3 = nn.LeakyReLU()
        
        # Convolutional block 4
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.act4 = nn.LeakyReLU()
        
        # Convolutional block 5
        self.conv5 = nn.Conv2d(16, 16, kernel_size=2, stride=2)
        self.bn5 = nn.BatchNorm2d(16)
        self.act5 = nn.LeakyReLU()
        
        # Fully connected layers
        # For (1, 128, 128) input, final size is (16, 4, 4) => 16 * 4 * 4 = 256
        self.fc1   = nn.Linear(16 * (height // 32) * (width // 32), 100)
        self.act6  = nn.LeakyReLU()
        self.num_classes = num_classes
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.act6(self.fc1(x))
        logits = self.fc2(x)
        # logits = sigmoid(logits)
        return logits
    
    
class TinyCNN(nn.Module):
    """Fixed version - replaces 524k param FC layer with additional conv layers."""
    def __init__(self, input_shape, num_classes=2):
        super(TinyCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 128→64
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            # Block 2: 64→32
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            
            # Block 3: 32→16
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            
            # ADD: Block 4: 16→8 (critical addition!)
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            
            # ADD: Block 5: 8→4 (even more reduction!)
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            
            # Global Average Pooling instead of flatten
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Tiny classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(48, num_classes)  # 48→2 instead of 8192→64!
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
    
    

class SCNN(nn.Module):
    def __init__(self, input_shape, num_classes=4):
        super(SCNN, self).__init__()
        
        # Parse (channels, height, width) from input_shape
        in_channels = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]

        # Convolutional block 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=8, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.ln1  = nn.LayerNorm([8, height // 2, width // 2])
        self.act1 = nn.LeakyReLU()
        
        # Convolutional block 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.ln2  = nn.LayerNorm([16, height // 4, width // 4])
        self.act2 = nn.LeakyReLU()
        
        # Convolutional block 3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.ln3  = nn.LayerNorm([32, height // 8, width // 8])
        self.act3 = nn.LeakyReLU()
        
        # Convolutional block 4
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.ln4  = nn.LayerNorm([16, height // 16, width // 16])
        self.act4 = nn.LeakyReLU()
        
        # Convolutional block 5
        self.conv5 = nn.Conv2d(16, 16, kernel_size=2, stride=2)
        self.ln5  = nn.LayerNorm([16, height // 32, width // 32])
        self.act5 = nn.LeakyReLU()
        
        # Fully connected layers
        # For (1, 128, 128) input, final size is (16, 4, 4) => 16 * 4 * 4 = 256
        self.fc1   = nn.Linear(16 * (height // 32) * (width // 32), 100)
        self.act6  = nn.LeakyReLU()
        self.num_classes = num_classes
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.act1(self.ln1(self.conv1(x)))
        x = self.act2(self.ln2(self.conv2(x)))
        x = self.act3(self.ln3(self.conv3(x)))
        x = self.act4(self.ln4(self.conv4(x)))
        x = self.act5(self.ln5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.act6(self.fc1(x))
        logits = self.fc2(x)
        return torch.softmax(logits, dim=1)


##########################################################
######### Residual Network Classifiers ###################
##########################################################

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act   = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.down = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)

class ResNet(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super().__init__()
        in_channels = input_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # Could try with convolutional layer here instead of maxpool
            #ResidualBlock(32,  32, stride=1),

            ResidualBlock(32,  64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128,256, stride=2),
            ResidualBlock(256,256, stride=2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc      = nn.Linear(256, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

class ScatterResNet(nn.Module):
    def __init__(self, scat_shape, num_classes=2):
        super().__init__()
        # scat_shape is (C_s, H_s, W_s)
        in_ch = scat_shape[0]
        self.features = nn.Sequential(
            ResidualBlock(in_ch,  32, stride=1),
            ResidualBlock(32,  64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128,256, stride=2),
        )
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc      = nn.Linear(256, num_classes)

    def forward(self, scat):
        # scat: scattering coefficients [B, C_s, H_s, W_s]
        x = self.features(scat)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


############################################################
## Dual Input Squeeze-and-Excitation Network Classifiers ###
############################################################


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

class ScatterSqueezeNet(nn.Module): #DualSSN
    def __init__(self, img_shape, scat_shape, num_classes,
                 hidden_dim1=128, hidden_dim2=64, classifier_hidden_dim=128,
                 dropout_rate=0.5, J=2):
        super(ScatterSqueezeNet, self).__init__()
        C_img, H_img, W_img = img_shape
        C_scat, H_scat, W_scat = scat_shape

        # ----------------------
        # Image Branch Encoder (same as DualClassifier)
        # ----------------------
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(C_img, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        self.conv_to_latent_img = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )

        # -------------------------------
        # Scattering Branch with SE attention
        # -------------------------------
        def conv_block(in_ch, out_ch, k, s, p, dropout_p=0.2):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_p)
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

        # Build scattering branch with SE blocks
        scat_blocks = []
        # initial conv + SE
        scat_blocks.append(conv_block(C_scat, hidden_dim2, 3, 1, 1, dropout_p=0.2))
        scat_blocks.append(SEBlock(hidden_dim2))
        # repeated downsampling + SE
        for i in range(downsample_blocks):
            dropout_p = 0.2 + i * 0.1  # Increase dropout with depth
            scat_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 1, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim2))
            scat_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 2, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim2))

        self.conv_to_latent_scat = nn.Sequential(*scat_blocks)

        # ---------------------------
        # Compute combined feature size
        # ---------------------------
        with torch.no_grad():
            dummy_img  = torch.zeros(1, C_img, H_img, W_img)
            dummy_scat = torch.zeros(1, C_scat, H_scat, W_scat)
            img_f = self.cnn_encoder(dummy_img)
            img_f = self.conv_to_latent_img(img_f)
            scat_f = self.conv_to_latent_scat(dummy_scat)
            img_dim  = img_f.view(1, -1).size(1)
            scat_dim = scat_f.view(1, -1).size(1)
            combined_dim = img_dim + scat_dim

        # ----------------------
        # Classifier Head
        # ----------------------
        self.FC_input       = nn.Linear(combined_dim, hidden_dim1)
        self.bn1            = nn.BatchNorm1d(hidden_dim1)
        self.dropout1       = nn.Dropout(dropout_rate)
        self.FC_hidden      = nn.Linear(hidden_dim1, classifier_hidden_dim)
        self.bn2            = nn.BatchNorm1d(classifier_hidden_dim)
        self.dropout2       = nn.Dropout(dropout_rate)
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
        #return torch.softmax(self.FC_classifier(x), dim=1)  # Return probabilities for multi-class classification
        return self.FC_classifier(x)  # Return logits directly for multi-class classification
    

class TinyScatterSqueezeNet(nn.Module): # DualSSN
    def __init__(self, img_shape, scat_shape, num_classes,
                 hidden_dim1=32, hidden_dim2=16, classifier_hidden_dim=32,
                 dropout_rate=0.5, J=2):  # Increase default dropout to 0.5
        super(TinyScatterSqueezeNet, self).__init__()
        C_img, H_img, W_img = img_shape
        C_scat, H_scat, W_scat = scat_shape

        # Image Branch with Dropout
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(C_img, 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),  # ADD: Spatial dropout after first conv
            
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),  # ADD: Spatial dropout after second conv
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)  # ADD: Spatial dropout after third conv
        )
        
        self.conv_to_latent_img = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),  # ADD: Dropout in latent layers
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),  # ADD: Dropout in latent layers
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),  # ADD: Dropout in latent layers
            
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.4)  # ADD: Higher dropout near bottleneck
        )

        # Scattering branch conv_block helper
        def conv_block(in_ch, out_ch, k, s, p, dropout_p=0.2):  # MODIFY: Add dropout parameter
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout_p)  # ADD: Dropout in conv blocks
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
            dropout_p = 0.2 + i * 0.1  # Increase dropout as we go deeper
            scat_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 1, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim2))
            scat_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 2, 1, dropout_p=dropout_p))
            scat_blocks.append(SEBlock(hidden_dim2))

        self.conv_to_latent_scat = nn.Sequential(*scat_blocks)

        # Compute combined feature size (unchanged)
        with torch.no_grad():
            dummy_img  = torch.zeros(1, C_img, H_img, W_img)
            dummy_scat = torch.zeros(1, C_scat, H_scat, W_scat)
            img_f = self.cnn_encoder(dummy_img)
            img_f = self.conv_to_latent_img(img_f)
            scat_f = self.conv_to_latent_scat(dummy_scat)
            img_dim  = img_f.view(1, -1).size(1)
            scat_dim = scat_f.view(1, -1).size(1)
            combined_dim = img_dim + scat_dim

        # Classifier Head - CRITICAL: Increase dropout here
        self.FC_input       = nn.Linear(combined_dim, hidden_dim1)
        self.bn1            = nn.BatchNorm1d(hidden_dim1)
        self.dropout1       = nn.Dropout(0.5)  # MODIFY: Separate dropout layer
        self.FC_hidden      = nn.Linear(hidden_dim1, classifier_hidden_dim)
        self.bn2            = nn.BatchNorm1d(classifier_hidden_dim)
        self.dropout2       = nn.Dropout(0.5)  # MODIFY: Separate dropout layer
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
        # MLP head - MODIFY: Use separate dropout layers
        x = self.act(self.bn1(self.FC_input(x)))
        x = self.dropout1(x)  # MODIFY: Use self.dropout1
        x = self.act(self.bn2(self.FC_hidden(x)))
        x = self.dropout2(x)  # MODIFY: Use self.dropout2
        return self.FC_classifier(x)
    

# This classifier is similar to ScatterSqueezeNet but with SE blocks in the image branch as well.
class ScatterSqueezeNet2(nn.Module):
    def __init__(self, img_shape, scat_shape, num_classes,
                 hidden_dim1=256, hidden_dim2=128, classifier_hidden_dim=256,
                 dropout_rate=0.3, J=2):
        super(ScatterSqueezeNet2, self).__init__()
        C_img, H_img, W_img = img_shape
        C_scat, H_scat, W_scat = scat_shape

        # ----------------------
        # Image Branch Encoder (same as DualClassifier)
        # ----------------------
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(C_img, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            SEBlock(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            SEBlock(64),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            SEBlock(128)
        )

        self.conv_to_latent_img = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            SEBlock(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            SEBlock(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            SEBlock(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            SEBlock(128)
        )

        # -------------------------------
        # Scattering Branch with SE attention
        # -------------------------------
        def conv_block(in_ch, out_ch, k, s, p):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2)
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

        # Build scattering branch with SE blocks
        scat_blocks = []
        # initial conv + SE
        scat_blocks.append(conv_block(C_scat, hidden_dim2, 3, 1, 1))
        scat_blocks.append(SEBlock(hidden_dim2))
        # repeated downsampling + SE
        for _ in range(downsample_blocks):
            scat_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 1, 1))
            scat_blocks.append(SEBlock(hidden_dim2))
            scat_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 2, 1))
            scat_blocks.append(SEBlock(hidden_dim2))

        self.conv_to_latent_scat = nn.Sequential(*scat_blocks)

        # ---------------------------
        # Compute combined feature size
        # ---------------------------
        with torch.no_grad():
            dummy_img  = torch.zeros(1, C_img, H_img, W_img)
            dummy_scat = torch.zeros(1, C_scat, H_scat, W_scat)
            img_f = self.cnn_encoder(dummy_img)
            img_f = self.conv_to_latent_img(img_f)
            scat_f = self.conv_to_latent_scat(dummy_scat)
            img_dim  = img_f.view(1, -1).size(1)
            scat_dim = scat_f.view(1, -1).size(1)
            combined_dim = img_dim + scat_dim

        # ----------------------
        # Classifier Head
        # ----------------------
        self.FC_input       = nn.Linear(combined_dim, hidden_dim1)
        self.bn1            = nn.BatchNorm1d(hidden_dim1)
        self.FC_hidden      = nn.Linear(hidden_dim1, classifier_hidden_dim)
        self.bn2            = nn.BatchNorm1d(classifier_hidden_dim)
        self.FC_classifier  = nn.Linear(classifier_hidden_dim, num_classes)
        self.act            = nn.LeakyReLU(0.2)
        self.dropout        = nn.Dropout(dropout_rate)

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
        x = self.dropout(x)
        x = self.act(self.bn2(self.FC_hidden(x)))
        x = self.dropout(x)
        #return torch.softmax(self.FC_classifier(x), dim=1)  # Return probabilities for multi-class classification
        return self.FC_classifier(x)  # Return logits directly for multi-class classification
    

class DualCNNSqueezeNet_legacy(nn.Module):
    """
    Two-branch CNN classifier with SE attention in each branch,
    both processing the same input tensor but using global pooling
    to reduce dimensionality before the MLP head.
    """
    def __init__(self, input_shape, num_classes=4,
                 hidden_dim1=256, classifier_hidden_dim=256,
                 dropout_rate=0.3, reduction=16):
        super().__init__()
        C, H, W = input_shape

        def make_branch1():
            return nn.Sequential(
                nn.Conv2d(C, 8, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                SEBlock(8, reduction),
                nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                SEBlock(16, reduction),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                SEBlock(32, reduction),
                nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                SEBlock(16, reduction),
                nn.Conv2d(16, 16, kernel_size=2, stride=2),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                SEBlock(16, reduction),
                nn.AdaptiveAvgPool2d(1)
            )

        def make_branch2():
            return nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
                InceptionBlock(32),
                nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
                nn.AdaptiveAvgPool2d(1)
            )


        self.branch1 = make_branch1()
        self.branch2 = make_branch2()

        # Compute flattened feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            f1 = self.branch1(dummy)
            f2 = self.branch2(dummy)
            dim1 = f1.view(1, -1).size(1)
            dim2 = f2.view(1, -1).size(1)
            combined_dim = dim1 + dim2

        # Classifier head
        self.fc1        = nn.Linear(combined_dim, hidden_dim1)
        self.bn1        = nn.BatchNorm1d(hidden_dim1)
        self.fc2        = nn.Linear(hidden_dim1, classifier_hidden_dim)
        self.bn2        = nn.BatchNorm1d(classifier_hidden_dim)
        self.classifier = nn.Linear(classifier_hidden_dim, num_classes)
        self.act        = nn.LeakyReLU(0.2)
        self.dropout    = nn.Dropout(dropout_rate)

    def forward(self, x):
        x1 = self.branch1(x).view(x.size(0), -1)
        x2 = self.branch2(x).view(x.size(0), -1)
        x  = torch.cat([x1, x2], dim=1)
        x  = self.act(self.bn1(self.fc1(x)))
        x  = self.dropout(x)
        x  = self.act(self.bn2(self.fc2(x)))
        x  = self.dropout(x)
        return self.classifier(x)


class DualCNNSqueezeNet(nn.Module):
    """
    Dual-branch CNN directly analogous to TinyScatterSqueezeNet.
    
    Architecture mirrors DualSSN but processes images in both branches:
    - Branch 1 (img_branch): Mimics DualSSN's image encoder (5×5 and 3×3 kernels)
    - Branch 2 (detail_branch): Mimics DualSSN's scattering branch (all 3×3 kernels)
    
    Key differences from failed DualCSN:
    - 32 channels (not 16) to match DualSSN capacity
    - Separate encoder + latent compression like DualSSN
    - MLP classifier head with BatchNorm (not just Linear layers)
    """
    def __init__(self, input_shape, num_classes=2, hidden_dim1=32, classifier_hidden_dim=32):
        super(DualCNNSqueezeNet, self).__init__()
        
        in_channels, H, W = input_shape
        
        # ==========================================
        # Branch 1: Image Encoder (mimics DualSSN's cnn_encoder)
        # Uses larger kernels (5×5, 3×3, 5×5) for broad features
        # ==========================================
        self.img_encoder = nn.Sequential(
            # First conv: 5×5 kernel like DualSSN
            nn.Conv2d(in_channels, 8, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),
            
            # Second conv: 3×3 kernel
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.2),
            
            # Spatial reduction with MaxPool
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv: 5×5 kernel like DualSSN
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3)
        )
        
        # Compression to latent space (mimics conv_to_latent_img)
        self.img_to_latent = nn.Sequential(
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
        
        # ==========================================
        # Branch 2: Detail Encoder (mimics DualSSN's scattering branch)
        # Uses 3×3 kernels throughout with SE blocks
        # ==========================================
        # Helper function for conv blocks with dropout
        def conv_block(in_ch, out_ch, k, s, p, dropout_p=0.2):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout_p)
            )
        
        # Build detail branch with SE blocks (16 channels like DualSSN's hidden_dim2)
        hidden_dim2 = 16
        detail_blocks = []
        
        # Initial conv + SE
        detail_blocks.append(conv_block(in_channels, hidden_dim2, 3, 1, 1, dropout_p=0.2))
        detail_blocks.append(SEBlock(hidden_dim2))
        
        # Three downsampling stages (assuming 128×128 input)
        # This mirrors DualSSN's scattering branch structure for J=2
        for i in range(3):
            dropout_p = 0.2 + i * 0.1
            # Non-strided conv + SE
            detail_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 1, 1, dropout_p=dropout_p))
            detail_blocks.append(SEBlock(hidden_dim2))
            # Strided conv + SE (downsampling)
            detail_blocks.append(conv_block(hidden_dim2, hidden_dim2, 3, 2, 1, dropout_p=dropout_p))
            detail_blocks.append(SEBlock(hidden_dim2))
        
        self.detail_to_latent = nn.Sequential(*detail_blocks)
        
        # ==========================================
        # Compute combined feature dimensions
        # ==========================================
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, H, W)
            
            # Branch 1 output
            img_f = self.img_encoder(dummy_input)
            img_f = self.img_to_latent(img_f)
            img_dim = img_f.view(1, -1).size(1)
            
            # Branch 2 output
            detail_f = self.detail_to_latent(dummy_input)
            detail_dim = detail_f.view(1, -1).size(1)
            
            combined_dim = img_dim + detail_dim
        
        # ==========================================
        # Classifier Head (exactly like DualSSN)
        # ==========================================
        self.FC_input = nn.Linear(combined_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(0.5)
        
        self.FC_hidden = nn.Linear(hidden_dim1, classifier_hidden_dim)
        self.bn2 = nn.BatchNorm1d(classifier_hidden_dim)
        self.dropout2 = nn.Dropout(0.5)
        
        self.FC_classifier = nn.Linear(classifier_hidden_dim, num_classes)
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        # Branch 1: Image encoding path
        x_img = self.img_encoder(x)
        x_img = self.img_to_latent(x_img)
        
        # Branch 2: Detail encoding path
        x_detail = self.detail_to_latent(x)
        
        # Flatten both branches
        x_img = x_img.view(x_img.size(0), -1)
        x_detail = x_detail.view(x_detail.size(0), -1)
        
        # Concatenate features
        x = torch.cat([x_img, x_detail], dim=1)
        
        # MLP classifier head (exactly like DualSSN)
        x = self.act(self.bn1(self.FC_input(x)))
        x = self.dropout1(x)
        x = self.act(self.bn2(self.FC_hidden(x)))
        x = self.dropout2(x)
        
        return self.FC_classifier(x)