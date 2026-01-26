import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class ProjectModel(nn.Module):
    def __init__(self, input_shape, num_classes=4):
        super(ProjectModel, self).__init__()
        # Define basic convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Dummy input to calculate the output size after feature extraction
        dummy_input = torch.zeros(1, *input_shape)
        dummy_feature_output = self.feature_extractor(dummy_input)
        output_size = dummy_feature_output.view(dummy_feature_output.size(0), -1).shape[1]
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
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

class BinaryClassifier(nn.Module):
    def __init__(self, input_shape, num_classes=2):
        super(BinaryClassifier, self).__init__()

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
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x, activation: bool = False):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.act6(self.fc1(x))
        logits = self.fc2(x)
        return logits.squeeze(1)
    

if False:
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
    
    class BinaryClassifier(nn.Module):
        def __init__(self, input_shape, num_classes=2):
            super().__init__()
            in_channels = input_shape[0]

            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                ResidualBlock(32,  64, stride=2),
                ResidualBlock(64, 128, stride=2),
                ResidualBlock(128,256, stride=2),
                ResidualBlock(256,256, stride=2),
            )

            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc  = nn.Linear(256, 1)

        def forward(self, x):
            x = self.features(x)
            x = self.gap(x).view(x.size(0), -1)
            return self.fc(x)

else: 
    if False:
        class BinaryClassifier(nn.Module):
            def __init__(self, input_shape, num_classes=2):
                super(BinaryClassifier, self).__init__()

                # Parse (channels, height, width) from input_shape
                in_channels = input_shape[0]
                height = input_shape[1]
                width = input_shape[2]

                # Convolutional block 1
                self.conv1 = nn.Conv2d(
                    in_channels=in_channels, 
                    out_channels=32, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                )
                self.bn1 = nn.BatchNorm2d(32)    
                self.act1 = nn.LeakyReLU()
                
                # Convolutional block 2
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
                self.bn2 = nn.BatchNorm2d(64)
                self.act2 = nn.LeakyReLU()
                
                # Convolutional block 3
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
                self.bn3 = nn.BatchNorm2d(128)
                self.act3 = nn.LeakyReLU()
                
                # Convolutional block 4
                self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1)
                self.bn4 = nn.BatchNorm2d(64)
                self.act4 = nn.LeakyReLU()
                
                # Convolutional block 5
                self.conv5 = nn.Conv2d(64, 32, kernel_size=2, stride=2)
                self.bn5 = nn.BatchNorm2d(32)
                self.act5 = nn.LeakyReLU()
                
                # Fully connected layers
                # For (1, 128, 128) input, final size is (16, 4, 4) => 16 * 4 * 4 = 256
                self.fc1 = nn.Linear(32 * (height // 32) * (width // 32), 100)
                self.act6  = nn.LeakyReLU()
                self.num_classes = num_classes
                self.fc2 = nn.Linear(100, 1)

            def forward(self, x, activation: bool = False):
                x = self.act1(self.bn1(self.conv1(x)))
                x = self.act2(self.bn2(self.conv2(x)))
                x = self.act3(self.bn3(self.conv3(x)))
                x = self.act4(self.bn4(self.conv4(x)))
                x = self.act5(self.bn5(self.conv5(x)))
                x = x.view(x.size(0), -1)
                x = self.act6(self.fc1(x))
                logits = self.fc2(x)
                return logits.squeeze(1)
    else:

        class BinaryClassifier(nn.Module):
            def __init__(self, input_shape):
                super(BinaryClassifier, self).__init__()
                in_channels = input_shape[0]  # should be 1 for grayscale

                self.conv_layers = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),

                    nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),

                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    # collapse spatial dims to 1×1
                    nn.AdaptiveAvgPool2d((1, 1))
                )

                self.classifier = nn.Linear(128, 1)

            def forward(self, x):
                x = self.conv_layers(x)                 # [B,128,1,1]
                x = x.view(x.size(0), -1)               # [B,128]
                return self.classifier(x)               # [B,1]


class RustigeClassifier(nn.Module):
    def __init__(self, input_shape, num_classes=4):
        super(RustigeClassifier, self).__init__()
        
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

    def forward(self, x, activation: bool = False):
        x = self.act1(self.ln1(self.conv1(x)))
        x = self.act2(self.ln2(self.conv2(x)))
        x = self.act3(self.ln3(self.conv3(x)))
        x = self.act4(self.ln4(self.conv4(x)))
        x = self.act5(self.ln5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.act6(self.fc1(x))
        logits = self.fc2(x)
        return logits



# DANN classifier

class DANNClassifier(nn.Module):
    def __init__(self, input_shape, num_classes=4):
        super(DANNClassifier, self).__init__()
        
        # shared feature extractor (conv → norm → act → flatten → FC)
        C, H, W = input_shape
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(C, 8, 3, 2, 1),
            nn.LayerNorm([8, H//2,  W//2]),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.LayerNorm([16, H//4, W//4]),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.LayerNorm([32, H//8, W//8]),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3, 2, 1),
            nn.LayerNorm([16, H//16, W//16]),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, 2, 2),
            nn.LayerNorm([16, H//32, W//32]),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(16*(H//32)*(W//32), 100),
            nn.LeakyReLU()
        )

        # two heads: one for your galaxy‐class, one for real/fake domain
        self.classifier_head = nn.Linear(100, num_classes)
        self.domain_head     = nn.Linear(100, 2)


    def forward(self, x, alpha=1.0):
        feat = self.feature_extractor(x)                 # [B,100]
        class_logits  = self.classifier_head(feat)       # [B,num_classes]
        dom_feat      = grad_reverse(feat, lambd=alpha)  # GRL on feature
        domain_logits = self.domain_head(dom_feat)       # [B,2]
        return class_logits, domain_logits

# Scattering classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=4, hidden_dim=120):
        super(MLPClassifier, self).__init__()
        # Compute second layer dimension such that when hidden_dim=120, fc2 becomes 84.
        fc2_dim = int(hidden_dim * 0.7)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input if necessary
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x




class DualClassifier(nn.Module):
    def __init__(self, img_shape, scat_shape, num_classes, 
                 hidden_dim1=256, hidden_dim2=128, classifier_hidden_dim=256, 
                 dropout_rate=0.3, J=2):
        """
        Args:
            img_shape: Tuple (channels, height, width) of the full image.
            scat_shape: Tuple (channels, height, width) of the scattering coefficients.
            num_classes: Number of output classes.
            hidden_dim1, hidden_dim2: Dimensions for intermediate features.
            classifier_hidden_dim: Dimension for the hidden FC layer before classification.
            dropout_rate: Dropout probability.
            J: Determines the number of downsampling blocks for the scattering branch.
        """
        super(DualClassifier, self).__init__()

        # ----------------------
        # Image Branch Encoder
        # ----------------------
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.conv_to_latent_img = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        # -------------------------------
        # Scattering Coefficients Branch
        # -------------------------------
        # Define a helper function for a conv block.
        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        
        # Determine the number of downsampling blocks based on J.
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
        
        # Build the scattering branch.
        scat_in_channels = scat_shape[0]
        conv_blocks = []
        for _ in range(downsample_blocks):
            conv_blocks.append(conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=1, padding=1))
            conv_blocks.append(conv_block(hidden_dim2, hidden_dim2, kernel_size=3, stride=2, padding=1))
        
        self.conv_to_latent_scat = nn.Sequential(
            conv_block(scat_in_channels, hidden_dim2, kernel_size=3, stride=1, padding=1),
            *conv_blocks
        )
        
        # -------------------------------------------
        # Determine Combined Feature Dimensions
        # -------------------------------------------
        # Use dummy inputs to compute the flattened dimensions from both branches.
        with torch.no_grad():
            dummy_img = torch.zeros(1, *img_shape)
            dummy_scat = torch.zeros(1, *scat_shape)
            img_feat = self.cnn_encoder(dummy_img)
            img_feat = self.conv_to_latent_img(img_feat)
            scat_feat = self.conv_to_latent_scat(dummy_scat)
            img_flat_dim = img_feat.view(1, -1).size(1)
            scat_flat_dim = scat_feat.view(1, -1).size(1)
            combined_dim = img_flat_dim + scat_flat_dim
        
        # ----------------------
        # Fully Connected Layers
        # ----------------------
        self.FC_input = nn.Linear(combined_dim, 384, bias=True)
        self.bn1 = nn.BatchNorm1d(384)
        self.FC_hidden = nn.Linear(384, classifier_hidden_dim, bias=True)
        self.bn2 = nn.BatchNorm1d(classifier_hidden_dim)
        self.FC_classifier = nn.Linear(classifier_hidden_dim, num_classes, bias=True)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, img, scat):
        # Process the image branch.
        cnn_features = self.cnn_encoder(img)
        img_features = self.conv_to_latent_img(cnn_features)
        
        # Process the scattering branch.
        scat_features = self.conv_to_latent_scat(scat)
        
        # Concatenate along the channel dimension.
        # (Assuming that the spatial dimensions of img_features and scat_features match.)
        combined = torch.cat([img_features, scat_features], dim=1)
        combined = combined.view(combined.size(0), -1)  # Flatten
        
        # Pass through fully connected layers.
        h = self.FC_input(combined)
        h = self.LeakyReLU(self.bn1(h))
        h = self.dropout(h)
        h = self.FC_hidden(h)
        h = self.LeakyReLU(self.bn2(h))
        h = self.dropout(h)
        logits = self.FC_classifier(h)
        return logits

