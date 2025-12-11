import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicMSELoss(nn.Module):
    def __init__(self, threshold=0.0):
        super(BasicMSELoss, self).__init__()
        self.threshold = threshold

    def forward(self, input, target, return_map=False):
        # Create mask based on the threshold (default is 0)
        mask = (target > self.threshold).float()

        # Calculate the MSE losses for masked and unmasked elements
        mse_loss_non_zero = (input * mask - target * mask).pow(2)
        mse_loss_zero = (input * (1 - mask) - target * (1 - mask)).pow(2)

        # Combine the losses
        loss = mse_loss_non_zero + mse_loss_zero

        if return_map:
            return loss, mask
        else:
            return loss.mean()
        
        
class NormalisedWeightedMSELoss(nn.Module):
    """A custom loss function that calculates the MSE loss with the sum of the relevant pixels weighted by a factor"""
    def __init__(self, threshold=0.1, weight=0.001):
        super(NormalisedWeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, input, target, return_map=False):
        # Create mask and initial weight map
        mask = (target > self.threshold).float()
        weight_map = torch.ones_like(target)
        num_masked_elements = mask.sum()
        num_unmasked_elements = mask.numel() - num_masked_elements

        # Normalize the weight_map to ensure the total weight of the region above the threshold is 10 times the total weight of the rest
        if num_unmasked_elements > 0:
            total_weight_masked, total_weight_unmasked = self.weight, 1
            weight_map[mask == 1] = total_weight_masked / num_masked_elements
            weight_map[mask == 0] = total_weight_unmasked / num_unmasked_elements

        # Normalize so that the sum of the weight map is 1
        weight_map = weight_map / weight_map.sum()

        # Calculate loss
        loss = nn.functional.mse_loss(input, target, reduction='none') * weight_map
        
        if return_map:
            return loss, weight_map
        else:
            return loss.sum()




class RadialWeightedMSELoss(nn.Module):
    """A custom loss function that calculates the MSE loss with the sum of the relevant pixels weighted by a factor and adds a radial weight"""
    def __init__(self, threshold=0.1, intensity_weight=0.001, radial_weight=0.001):
        super(RadialWeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.intensity_weight = intensity_weight
        self.radial_weight = radial_weight

    def forward(self, input, target, return_map=False):
        # Ensure input and target are 4D tensors
        if input.dim() == 3:
            input = input.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Create mask and map
        mask = (target > self.threshold).float()
        intensity_map = (target * mask) ** self.intensity_weight

        # Calculate radial distance map
        _, _, height, width = target.shape
        y_center, x_center = height // 2, width // 2
        y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        y_grid = y_grid.to(target.device).float()
        x_grid = x_grid.to(target.device).float()
        distance_map = torch.sqrt((x_grid - x_center) ** 2 + (y_grid - y_center) ** 2)
        max_distance = torch.max(distance_map)
        radial_weight_map = (1.0 - (distance_map / max_distance)) ** self.radial_weight

        weight_map = intensity_map * radial_weight_map
        loss_map = nn.functional.mse_loss(input, target, reduction='none') * weight_map
        zero_pixels_loss_map = nn.functional.mse_loss(input * (1 - mask), target * (1 - mask), reduction='none')

        total_loss_map = loss_map + zero_pixels_loss_map
        
        if return_map:
            return total_loss_map, weight_map
        else:
            return total_loss_map.sum()


class CustomIntensityWeightedMSELoss(nn.Module):
    """Weighting the intensity distribution too"""
    def __init__(self, intensity_threshold=0.1, intensity_weight=0.0001, log_weight=0.0001):
        super(CustomIntensityWeightedMSELoss, self).__init__()
        self.intensity_threshold = intensity_threshold
        self.intensity_weight = intensity_weight
        self.log_weight = log_weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, generated, target, return_map=False):
        # Ensure that generated and target have the correct dimensions
        if generated.dim() == 3:
            generated = generated.unsqueeze(1)  # Add a channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add a channel dimension

        # Compute base MSE loss
        mse_loss = self.mse_loss(generated, target)

        # Apply intensity weighting
        intensity_mask = (target > self.intensity_threshold).float()
        weighted_mse_loss = mse_loss * (1 + intensity_mask * (self.intensity_weight - 1))

        # Apply logarithmic weighting
        log_diff = torch.abs(torch.log1p(generated) - torch.log1p(target))
        log_loss = self.log_weight * log_diff

        # Combine losses
        total_loss = weighted_mse_loss + log_loss

        if return_map:
            # Ensure that the loss map has the correct dimensions
            return total_loss, intensity_mask
        else:
            return total_loss.mean()

    

class WeightedMSELoss(nn.Module):
    """Weighting pixels above threshold more heavily"""
    def __init__(self, threshold=0.1, weight=0.001):
        super(WeightedMSELoss, self).__init__()
        self.threshold = threshold
        self.intensity_weight = weight

    def forward(self, input, target, return_map=False):
        # Ensure that the input and target have the correct dimensions
        if input.dim() == 3:
            input = input.unsqueeze(1)  # Add a channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add a channel dimension

        mask = (target > self.threshold).float()
        intensity_map = (target * mask) ** self.intensity_weight
        
        # Compute the MSE loss for the masked regions
        mse_loss = nn.functional.mse_loss(input * mask * intensity_map, target * mask * intensity_map, reduction='none')
        zero_pixels_loss = nn.functional.mse_loss(input * (1 - mask), target * (1 - mask), reduction='none')

        # Combine the losses
        total_loss = mse_loss + zero_pixels_loss

        if return_map:
            # Ensure the loss map has the correct dimensions
            return total_loss, intensity_map
        else:
            return total_loss.sum()



class MaxIntensityMSELoss(nn.Module):
    def __init__(self, intensity_weight=0.001, sum_weight=0.001):
        super(MaxIntensityMSELoss, self).__init__()
        self.intensity_weight = intensity_weight
        self.sum_weight = sum_weight
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, input, target, return_map=False):
        # Calculate the standard MSE loss element-wise
        loss_map = self.mse_loss(input, target)
        weight_map = torch.ones_like(loss_map)

        # Calculate the difference of the maximum intensity pixels
        max_intensity_input = input.view(input.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_target = target.view(target.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_diff = torch.abs(max_intensity_input - max_intensity_target)

        # Calculate the difference of the summed pixel intensities
        sum_intensity_input = input.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_target = target.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_diff = torch.abs(sum_intensity_input - sum_intensity_target)

        # Add these terms to the loss
        max_intensity_penalty = self.intensity_weight * max_intensity_diff.mean()
        sum_intensity_penalty = self.sum_weight * sum_intensity_diff.mean()

        # Scale down the additional penalties
        combined_loss = loss_map + max_intensity_penalty + sum_intensity_penalty

        if return_map:
            return combined_loss, weight_map
        else:
            return combined_loss.sum()



class CustomMSELoss(nn.Module):
    def __init__(self, intensity_weight=0.001, sum_weight=0.001):
        super(CustomMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.intensity_weight = intensity_weight
        self.sum_weight = sum_weight
        self.epsilon = 1e-8  # Small constant to prevent division by zero

    def forward(self, input, target, return_map=False):
        # Calculate the standard MSE loss element-wise
        loss_map = self.mse_loss(input, target)
        weight_map = torch.ones_like(loss_map)
        weighted_loss_map = loss_map * weight_map
        
        # Calculate the root mean squared difference of the maximum intensity pixels
        max_intensity_input = input.view(input.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_target = target.view(target.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_diff = torch.sqrt(torch.abs(max_intensity_input - max_intensity_target) + self.epsilon).view(-1, 1, 1, 1)

        # Calculate the root mean squared difference of the summed pixel intensities
        sum_intensity_input = input.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_target = target.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_diff = torch.sqrt(torch.abs(sum_intensity_input - sum_intensity_target) + self.epsilon).view(-1, 1, 1, 1)

        # Normalize the penalties relative to the input size
        num_elements = torch.numel(input[0]) + self.epsilon  # Prevent division by zero
        max_intensity_diff = max_intensity_diff / num_elements
        sum_intensity_diff = sum_intensity_diff / num_elements

        # Expand the differences to match the loss map's shape
        max_intensity_diff = max_intensity_diff.expand_as(loss_map)
        sum_intensity_diff = sum_intensity_diff.expand_as(loss_map)

        # Combine the losses
        weighted_loss_map = weighted_loss_map + \
                            self.intensity_weight * max_intensity_diff + \
                            self.sum_weight * sum_intensity_diff

        if return_map:
            return weighted_loss_map, weight_map
        else:
            return weighted_loss_map.sum()

        
class StandardMSELoss(nn.Module):
    def __init__(self):
        super(StandardMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, input, target, return_map=False):
        if return_map:
            # Compute the loss per pixel
            loss_map = self.mse_loss(input, target)
            weight_map = torch.ones_like(loss_map)
            weighted_loss_map = loss_map * weight_map
            return weighted_loss_map, weight_map
        else:
            # Return the sum of the loss across all pixels
            return self.mse_loss(input, target).sum()
    

class CombinedMSELoss(nn.Module):
    def __init__(self, intensity_weight=0.001, sum_weight=0.001, threshold=0.1, high_intensity_weight=0.001, epsilon=1e-8):
        super(CombinedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.intensity_weight = intensity_weight
        self.sum_weight = sum_weight
        self.threshold = threshold
        self.high_intensity_weight = high_intensity_weight
        self.epsilon = epsilon

    def forward(self, input, target, return_map=False):
        # Ensure that the input and target have the correct dimensions
        if input.dim() == 3:
            input = input.unsqueeze(1)  # Add a channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add a channel dimension

        # Create mask and intensity map for high-intensity weighting
        mask = (target > self.threshold).float()
        intensity_map = torch.clamp(target * mask, min=self.epsilon) ** self.high_intensity_weight

        # Compute the MSE loss for the masked and unmasked regions
        target_loss = nn.functional.mse_loss(input * mask * intensity_map, target * mask * intensity_map, reduction='none')
        zero_pixels_loss = nn.functional.mse_loss(input * (1 - mask), target * (1 - mask), reduction='none')
        total_loss = target_loss + zero_pixels_loss

        # Root mean squared difference of the maximum intensity pixels
        max_intensity_input = input.view(input.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_target = target.view(target.size(0), -1).max(dim=-1, keepdim=True)[0]
        #max_intensity_diff = torch.sqrt(torch.clamp((max_intensity_input - max_intensity_target) ** 2, min=self.epsilon)).view(-1, 1, 1, 1)
        max_intensity_diff = torch.abs(max_intensity_input - max_intensity_target).view(-1, 1, 1, 1)


        # Root mean squared difference of the summed pixel intensities
        sum_intensity_input = input.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_target = target.sum(dim=[-1, -2], keepdim=True)
        #sum_intensity_diff = torch.sqrt(torch.clamp((sum_intensity_input - sum_intensity_target) ** 2, min=self.epsilon)).view(-1, 1, 1, 1)
        sum_intensity_diff = torch.abs(sum_intensity_input - sum_intensity_target).view(-1, 1, 1, 1)


        # Normalize the penalties relative to the input size
        max_intensity_diff = max_intensity_diff / (torch.numel(input[0]) + self.epsilon)
        sum_intensity_diff = sum_intensity_diff / (torch.numel(input[0]) + self.epsilon)

        # Expand the differences to match the loss map's shape
        max_intensity_diff = max_intensity_diff.expand_as(total_loss)
        sum_intensity_diff = sum_intensity_diff.expand_as(total_loss)

        # Combine the pixel-wise loss with the non-spatial terms
        total_loss = target_loss + zero_pixels_loss + \
                     self.intensity_weight * max_intensity_diff + \
                     self.sum_weight * sum_intensity_diff

        if return_map:
            return total_loss, intensity_map
        else:
            return total_loss.sum()



class ExperimentalMSELoss(nn.Module):
    def __init__(self, intensity_weight=0.001, sum_weight=0.001, histogram_weight=0.001, threshold=0.1, high_intensity_weight=0.001, epsilon=1e-8):
        super(ExperimentalMSELoss, self).__init__()
        self.intensity_weight = intensity_weight
        self.sum_weight = sum_weight
        self.histogram_weight = histogram_weight
        self.threshold = threshold
        self.high_intensity_weight = high_intensity_weight
        self.epsilon = epsilon
        self.num_bins = 10  # Number of histogram bins

    def calculate_differentiable_histogram(self, image, num_bins):
        # Compute a differentiable histogram using binning
        batch_size, num_pixels = image.size(0), image.size(1) * image.size(2) * image.size(3)
        image_flat = image.view(batch_size, -1)
        histograms = []

        for i in range(num_bins):
            lower_bound = i / num_bins
            upper_bound = (i + 1) / num_bins
            bin_mask = ((image_flat >= lower_bound) & (image_flat < upper_bound)).float()
            histogram = bin_mask.sum(dim=1) / num_pixels  # Normalize the histogram
            histograms.append(histogram)

        return torch.stack(histograms, dim=1)  # Return histogram with shape [batch_size, num_bins]

    def forward(self, input, target, return_map=False):
        # Ensure that the input and target have the correct dimensions
        if input.dim() == 3:
            input = input.unsqueeze(1)  # Add a channel dimension
        if target.dim() == 3:
            target = target.unsqueeze(1)  # Add a channel dimension

        # Create mask and intensity map for high-intensity weighting
        mask = (target > self.threshold).float()
        intensity_map = torch.clamp(target * mask, min=self.epsilon) ** self.high_intensity_weight

        # Compute the MSE loss for the masked and unmasked regions
        target_loss = F.mse_loss(input * mask * intensity_map, target * mask * intensity_map, reduction='none')
        zero_pixels_loss = F.mse_loss(input * (1 - mask), target * (1 - mask), reduction='none')
        total_loss = target_loss + zero_pixels_loss

        # Root mean squared difference of the maximum intensity pixels
        max_intensity_input = input.view(input.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_target = target.view(target.size(0), -1).max(dim=-1, keepdim=True)[0]
        max_intensity_diff = torch.abs(max_intensity_input - max_intensity_target).view(-1, 1, 1, 1)

        # Root mean squared difference of the summed pixel intensities
        sum_intensity_input = input.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_target = target.sum(dim=[-1, -2], keepdim=True)
        sum_intensity_diff = torch.abs(sum_intensity_input - sum_intensity_target).view(-1, 1, 1, 1)

        # Normalize the penalties relative to the input size
        max_intensity_diff = max_intensity_diff / (torch.numel(input[0]) + self.epsilon)
        sum_intensity_diff = sum_intensity_diff / (torch.numel(input[0]) + self.epsilon)

        # Calculate differentiable histograms for input and target
        input_histogram = self.calculate_differentiable_histogram(input, self.num_bins)
        target_histogram = self.calculate_differentiable_histogram(target, self.num_bins)

        # Histogram matching loss (e.g., L2 loss between histograms)
        histogram_loss = F.mse_loss(input_histogram, target_histogram, reduction='mean')

        # Combine the pixel-wise loss with the non-spatial terms and histogram loss
        # First, sum the pixel-wise total_loss
        total_loss = total_loss.sum()  # Now total_loss is a scalar

        # Add the scalar losses (intensity, sum intensity, and histogram losses)
        total_loss = total_loss + \
                     self.intensity_weight * max_intensity_diff.sum() + \
                     self.sum_weight * sum_intensity_diff.sum() + \
                     self.histogram_weight * histogram_loss

        if return_map:
            return total_loss, intensity_map
        else:
            return total_loss