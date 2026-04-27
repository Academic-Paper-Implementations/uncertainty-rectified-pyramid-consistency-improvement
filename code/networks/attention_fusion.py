import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleAttentionFusion(nn.Module):
    """
    Multi-Scale Attention Fusion Module
    Learns adaptive weights for each scale based on features and uncertainty maps.
    
    Instead of naive averaging: preds = (p1 + p2 + p3 + p4) / 4
    We compute: preds = w1*p1 + w2*p2 + w3*p3 + w4*p4, where sum(wi) = 1
    
    The weights are learned based on:
    1. The prediction confidence (entropy-based)
    2. Feature consistency across scales
    """
    
    def __init__(self, num_classes, num_scales=4, use_uncertainty=True):
        super(MultiScaleAttentionFusion, self).__init__()
        self.num_classes = num_classes
        self.num_scales = num_scales
        self.use_uncertainty = use_uncertainty
        
        # Attention network: takes concatenated predictions and outputs scale weights
        # Input: num_scales * num_classes channels (concatenated softmax outputs)
        # Output: num_scales channels (one weight per scale)
        
        self.attention_net = nn.Sequential(
            nn.Conv2d(num_scales * num_classes, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_scales, kernel_size=1, bias=True),
        )
        
        # Optional: Uncertainty-guided attention refinement
        if use_uncertainty:
            self.uncertainty_refine = nn.Sequential(
                nn.Conv2d(num_scales, num_scales, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_scales),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_scales, num_scales, kernel_size=1, bias=True),
            )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def compute_entropy(self, prob):
        """
        Compute entropy-based uncertainty for each prediction.
        Higher entropy = higher uncertainty = lower weight
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        entropy = -torch.sum(prob * torch.log(prob + eps), dim=1, keepdim=True)
        # Normalize entropy to [0, 1]
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=prob.dtype, device=prob.device))
        normalized_entropy = entropy / max_entropy
        return normalized_entropy
    
    def forward(self, predictions_soft):
        """
        Args:
            predictions_soft: list of softmax predictions from different scales
                              Each has shape [B, num_classes, H, W]
        
        Returns:
            fused_pred: weighted fusion of predictions [B, num_classes, H, W]
            attention_weights: learned weights for each scale [B, num_scales, H, W]
        """
        assert len(predictions_soft) == self.num_scales, \
            f"Expected {self.num_scales} predictions, got {len(predictions_soft)}"
        
        # Ensure all predictions have the same spatial size
        target_size = predictions_soft[0].shape[2:]
        aligned_preds = []
        for pred in predictions_soft:
            if pred.shape[2:] != target_size:
                pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=True)
            aligned_preds.append(pred)
        
        # Concatenate all predictions: [B, num_scales * num_classes, H, W]
        concat_preds = torch.cat(aligned_preds, dim=1)
        
        # Compute attention weights: [B, num_scales, H, W]
        attention_logits = self.attention_net(concat_preds)
        
        # Optionally refine with uncertainty information
        if self.use_uncertainty:
            # Compute entropy (uncertainty) for each scale
            entropies = []
            for pred in aligned_preds:
                entropy = self.compute_entropy(pred)
                entropies.append(entropy)
            # Stack entropies: [B, num_scales, H, W]
            uncertainty_maps = torch.cat(entropies, dim=1)
            
            # Use inverse uncertainty to refine attention
            # Lower uncertainty should have higher weight
            inverse_uncertainty = 1.0 - uncertainty_maps
            attention_logits = attention_logits + self.uncertainty_refine(inverse_uncertainty)
        
        # Apply softmax to get normalized weights (sum to 1)
        attention_weights = F.softmax(attention_logits, dim=1)
        
        # Weighted fusion
        fused_pred = torch.zeros_like(aligned_preds[0])
        for i, pred in enumerate(aligned_preds):
            # attention_weights[:, i:i+1, :, :] has shape [B, 1, H, W]
            fused_pred = fused_pred + attention_weights[:, i:i+1, :, :] * pred
        
        return fused_pred, attention_weights


class LightweightAttentionFusion(nn.Module):
    """
    Lightweight version using global average pooling for efficiency.
    Produces channel-wise (per-scale) weights instead of spatial weights.
    """
    
    def __init__(self, num_classes, num_scales=4):
        super(LightweightAttentionFusion, self).__init__()
        self.num_classes = num_classes
        self.num_scales = num_scales
        
        # Global attention network
        self.attention_net = nn.Sequential(
            nn.Linear(num_scales * num_classes, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_scales),
        )
        
    def forward(self, predictions_soft):
        """
        Args:
            predictions_soft: list of softmax predictions from different scales
        
        Returns:
            fused_pred: weighted fusion of predictions
            attention_weights: learned weights for each scale [B, num_scales]
        """
        # Ensure all predictions have the same spatial size
        target_size = predictions_soft[0].shape[2:]
        aligned_preds = []
        for pred in predictions_soft:
            if pred.shape[2:] != target_size:
                pred = F.interpolate(pred, size=target_size, mode='bilinear', align_corners=True)
            aligned_preds.append(pred)
        
        # Global average pooling on each prediction
        pooled_features = []
        for pred in aligned_preds:
            # [B, num_classes, H, W] -> [B, num_classes]
            pooled = F.adaptive_avg_pool2d(pred, 1).view(pred.size(0), -1)
            pooled_features.append(pooled)
        
        # Concatenate: [B, num_scales * num_classes]
        concat_features = torch.cat(pooled_features, dim=1)
        
        # Compute attention weights: [B, num_scales]
        attention_weights = F.softmax(self.attention_net(concat_features), dim=1)
        
        # Weighted fusion
        fused_pred = torch.zeros_like(aligned_preds[0])
        for i, pred in enumerate(aligned_preds):
            # Expand weights to [B, 1, 1, 1] for broadcasting
            weight = attention_weights[:, i:i+1].unsqueeze(-1).unsqueeze(-1)
            fused_pred = fused_pred + weight * pred
        
        return fused_pred, attention_weights
