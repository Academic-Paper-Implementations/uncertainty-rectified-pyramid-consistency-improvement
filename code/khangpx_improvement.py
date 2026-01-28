"""
URPC with Boundary-Aware Loss (SDM Integration)
===============================================
Author: KhangPX
Based on: train_uncertainty_rectified_pyramid_consistency_2D.py

Improvement: Tích hợp Signed Distance Map (SDM) để tạo Boundary-Aware Loss,
phạt nặng các pixel dự đoán sai ở gần ranh giới đối tượng.
"""

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import distance_transform_edt
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from utils import losses, metrics, ramps
from val_2D import test_single_volume_ds
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/URPC_Boundary_Aware', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_urpc', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# ============ BOUNDARY-AWARE LOSS PARAMETERS (NEW) ============
parser.add_argument('--boundary_weight', type=float, default=1.0,
                    help='weight for boundary-aware loss')
parser.add_argument('--sdm_sigma', type=float, default=5.0,
                    help='sigma for SDM weight normalization (controls boundary width)')
# ==============================================================

args = parser.parse_args()


# ============ SIGNED DISTANCE MAP (SDM) FUNCTIONS - OPTIMIZED ============
# Cải tiến 1: Tối ưu hóa hiệu suất tính toán SDM
# - Giảm thiểu chuyển đổi CPU/GPU bằng cách batch processing
# - Cache kết quả SDM để tái sử dụng trong cùng iteration

# Global cache để lưu SDM đã tính (tránh tính lại trong cùng iteration)
_sdm_cache = {}


def compute_sdm_single_class(binary_mask):
    """
    Tính Signed Distance Map cho một binary mask.
    Tối ưu: Sử dụng numpy operations hiệu quả hơn.
    
    Args:
        binary_mask: numpy array (H, W), giá trị 0 hoặc 1
        
    Returns:
        sdm: numpy array (H, W), Signed Distance Map
    """
    # Chuyển về uint8 một lần duy nhất
    binary_mask = np.ascontiguousarray(binary_mask, dtype=np.uint8)
    
    # Trường hợp đặc biệt: mask rỗng hoặc đầy
    foreground_sum = binary_mask.sum()
    if foreground_sum == 0:
        # Không có foreground -> distance âm từ edge
        return -distance_transform_edt(np.ones_like(binary_mask)).astype(np.float32)
    if foreground_sum == binary_mask.size:
        # Tất cả là foreground -> distance dương từ edge
        return distance_transform_edt(binary_mask).astype(np.float32)
    
    # Tính SDM: dương bên trong, âm bên ngoài
    # Pixel ở boundary có |SDM| ≈ 0
    dist_inside = distance_transform_edt(binary_mask)
    dist_outside = distance_transform_edt(1 - binary_mask)
    
    return (dist_inside - dist_outside).astype(np.float32)


def compute_sdm_batch_optimized(labels, num_classes, device):
    """
    Tính SDM cho batch labels - Phiên bản tối ưu.
    
    Cải tiến:
    - Chỉ chuyển dữ liệu CPU->GPU một lần duy nhất sau khi tính xong toàn bộ batch
    - Sử dụng contiguous array để tăng tốc độ truy cập bộ nhớ
    - Pre-allocate numpy array để giảm memory allocation
    
    Args:
        labels: torch.Tensor (B, H, W) trên GPU
        num_classes: số lượng classes
        device: torch.device để đưa kết quả về
        
    Returns:
        sdm_batch: torch.Tensor (B, num_classes, H, W) trên device
    """
    # Chuyển labels về CPU một lần duy nhất (unavoidable với scipy)
    batch_size, height, width = labels.shape
    labels_np = labels.detach().cpu().numpy()
    
    # Pre-allocate output array (contiguous memory)
    sdm_batch = np.zeros((batch_size, num_classes, height, width), dtype=np.float32)
    
    # Tính SDM cho từng sample và class
    for b in range(batch_size):
        label_b = labels_np[b]
        for c in range(num_classes):
            # Tạo binary mask và tính SDM
            binary_mask = (label_b == c)
            sdm_batch[b, c] = compute_sdm_single_class(binary_mask)
    
    # Chuyển về GPU một lần duy nhất với non_blocking để overlap transfer
    return torch.from_numpy(sdm_batch).to(device, non_blocking=True)


def compute_boundary_weight_map_optimized(labels, num_classes, sigma=5.0):
    """
    Tạo weight map từ SDM - Phiên bản tối ưu.
    
    Công thức: weight = 1 + exp(-min|SDM|^2 / (2 * sigma^2))
    Tham khảo: Kervadec et al. "Boundary loss for highly unbalanced segmentation"
    
    Args:
        labels: torch.Tensor (B, H, W) trên GPU
        num_classes: số classes
        sigma: điều khiển độ rộng vùng boundary
        
    Returns:
        weight_map: torch.Tensor (B, H, W), range [1, 2]
        sdm: torch.Tensor (B, C, H, W) - trả về để tái sử dụng
    """
    device = labels.device
    
    # Kiểm tra kích thước đầu vào
    assert labels.dim() == 3, f"Labels phải có shape (B, H, W), got {labels.shape}"
    
    # Tính SDM cho tất cả classes
    sdm = compute_sdm_batch_optimized(labels, num_classes, device)  # (B, C, H, W)
    
    # Kiểm tra shape output
    B, H, W = labels.shape
    assert sdm.shape == (B, num_classes, H, W), f"SDM shape mismatch: {sdm.shape}"
    
    # Tính weight map trên GPU (tất cả operations đều trên GPU)
    sdm_abs = torch.abs(sdm)  # (B, C, H, W)
    min_sdm_abs, _ = torch.min(sdm_abs, dim=1)  # (B, H, W)
    
    # Gaussian-like decay từ boundary
    boundary_weight = torch.exp(-min_sdm_abs.pow(2) / (2 * sigma * sigma))
    weight_map = 1.0 + boundary_weight  # range [1, 2]
    
    return weight_map, sdm


class BoundaryAwareCELoss(nn.Module):
    """
    Boundary-Aware Cross Entropy Loss - Phiên bản tối ưu.
    
    Phạt nặng các pixel dự đoán sai ở gần ranh giới đối tượng.
    Tối ưu: Nhận weight_map từ bên ngoài để tránh tính toán SDM lặp lại.
    """
    
    def __init__(self, num_classes, sigma=5.0):
        super(BoundaryAwareCELoss, self).__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        self.ce_loss = CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, labels, weight_map=None):
        """
        Args:
            predictions: torch.Tensor (B, C, H, W), logits từ model
            labels: torch.Tensor (B, H, W), ground truth labels
            weight_map: torch.Tensor (B, H, W), pre-computed weight map (optional)
            
        Returns:
            loss: scalar, weighted cross entropy loss
        """
        # Kiểm tra shape consistency
        B, C, H, W = predictions.shape
        assert labels.shape == (B, H, W), f"Label shape mismatch: {labels.shape} vs ({B}, {H}, {W})"
        
        # Tính CE loss cho từng pixel
        ce_per_pixel = self.ce_loss(predictions, labels.long())  # (B, H, W)
        
        # Sử dụng weight_map nếu được cung cấp, không thì tính mới
        if weight_map is None:
            weight_map, _ = compute_boundary_weight_map_optimized(
                labels, self.num_classes, self.sigma
            )
        
        # Đảm bảo weight_map trên cùng device
        weight_map = weight_map.to(predictions.device)
        
        # Apply weight và normalize
        weighted_ce = ce_per_pixel * weight_map
        loss = weighted_ce.mean()
        
        return loss


class BoundaryAwareDiceLoss(nn.Module):
    """
    Weighted Dice Loss với trọng số biên - Viết lại theo công thức chuẩn.
    
    Tham khảo: Kervadec et al. "Boundary loss for highly unbalanced segmentation"
    
    Công thức Weighted Dice:
        Dice = 2 * sum(w * p * g) / (sum(w * p) + sum(w * g) + smooth)
    
    Trong đó:
        - w: weight map (cao ở biên, thấp ở vùng xa biên)
        - p: predicted probability
        - g: ground truth (one-hot)
    
    Cải tiến:
        - Nhân weight trực tiếp vào intersection và union
        - Đảm bảo loss nằm trong [0, 1]
        - Nhận pre-computed SDM để tái sử dụng
    """
    
    def __init__(self, num_classes, sigma=5.0, smooth=1e-5):
        super(BoundaryAwareDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        self.smooth = smooth
    
    def forward(self, predictions, labels, sdm=None):
        """
        Args:
            predictions: torch.Tensor (B, C, H, W), logits từ model
            labels: torch.Tensor (B, H, W), ground truth labels
            sdm: torch.Tensor (B, C, H, W), pre-computed SDM (optional)
            
        Returns:
            loss: scalar trong [0, 1], weighted dice loss
        """
        device = predictions.device
        B, C, H, W = predictions.shape
        
        # Kiểm tra shape
        assert labels.shape == (B, H, W), f"Label shape mismatch: {labels.shape}"
        assert C == self.num_classes, f"Num classes mismatch: {C} vs {self.num_classes}"
        
        # Softmax để có probabilities
        probs = torch.softmax(predictions, dim=1)  # (B, C, H, W)
        
        # One-hot encode labels
        labels_one_hot = torch.zeros_like(probs)
        labels_one_hot.scatter_(1, labels.unsqueeze(1).long(), 1)  # (B, C, H, W)
        
        # Tính hoặc sử dụng SDM đã có
        if sdm is None:
            sdm = compute_sdm_batch_optimized(labels, self.num_classes, device)
        else:
            sdm = sdm.to(device)
        
        # Kiểm tra SDM shape
        assert sdm.shape == (B, C, H, W), f"SDM shape mismatch: {sdm.shape}"
        
        # Tính weight map per-class từ SDM
        # Công thức: w = 1 + exp(-|SDM|^2 / (2*sigma^2))
        # Vùng gần biên (|SDM| nhỏ) -> weight cao (~2)
        # Vùng xa biên (|SDM| lớn) -> weight thấp (~1)
        sdm_abs = torch.abs(sdm)
        weight_map = 1.0 + torch.exp(-sdm_abs.pow(2) / (2 * self.sigma * self.sigma))  # (B, C, H, W)
        
        # ============ WEIGHTED DICE THEO CÔNG THỨC CHUẨN ============
        # Weighted intersection: sum(w * p * g)
        weighted_intersection = (weight_map * probs * labels_one_hot).sum(dim=(2, 3))  # (B, C)
        
        # Weighted union: sum(w * p) + sum(w * g)
        weighted_pred = (weight_map * probs).sum(dim=(2, 3))  # (B, C)
        weighted_target = (weight_map * labels_one_hot).sum(dim=(2, 3))  # (B, C)
        weighted_union = weighted_pred + weighted_target  # (B, C)
        
        # Weighted Dice per class
        dice_per_class = (2.0 * weighted_intersection + self.smooth) / (weighted_union + self.smooth)  # (B, C)
        
        # Bỏ qua background (class 0), average over foreground classes
        # dice_per_class[:, 1:] có shape (B, num_classes-1)
        mean_dice = dice_per_class[:, 1:].mean()
        
        # Dice Loss = 1 - Dice, đảm bảo trong [0, 1]
        dice_loss = torch.clamp(1.0 - mean_dice, min=0.0, max=1.0)
        
        return dice_loss


# ==================================================================


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1,
                        class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    
    # ============ LOSS FUNCTIONS ============
    # Standard losses (giữ nguyên từ baseline)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    
    # Boundary-Aware losses (CẢI TIẾN MỚI)
    boundary_ce_loss = BoundaryAwareCELoss(
        num_classes=num_classes, 
        sigma=args.sdm_sigma
    )
    boundary_dice_loss = BoundaryAwareDiceLoss(
        num_classes=num_classes, 
        sigma=args.sdm_sigma
    )
    # ========================================

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("Boundary-Aware Loss enabled with weight={}, sigma={}".format(
        args.boundary_weight, args.sdm_sigma))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    kl_distance = nn.KLDivLoss(reduction='none')
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = model(
                volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)

            # ============ CẢI TIẾN 3: TÁI CẤU TRÚC LOSS AGGREGATION ============
            # Chiến lược mới:
            # - Nhánh chính (outputs): Dùng Boundary-Aware Loss để học chi tiết biên
            # - Nhánh phụ (aux1,2,3): Dùng standard CE+Dice để ổn định vùng phân đoạn
            # ================================================================
            
            # Lấy labeled samples
            labeled_outputs = outputs[:args.labeled_bs]
            labeled_labels = label_batch[:args.labeled_bs]
            
            # --- Bước 1: Tính SDM và weight_map MỘT LẦN cho labeled batch ---
            # Tối ưu: Tránh tính lại SDM nhiều lần trong cùng iteration
            weight_map, sdm = compute_boundary_weight_map_optimized(
                labeled_labels, num_classes, args.sdm_sigma
            )
            
            # --- Bước 2: NHÁNH CHÍNH - Boundary-Aware Loss ---
            # Main output học chi tiết biên sắc nét
            loss_boundary_ce = boundary_ce_loss(
                labeled_outputs, 
                labeled_labels,
                weight_map=weight_map  # Tái sử dụng weight_map đã tính
            )
            loss_boundary_dice = boundary_dice_loss(
                labeled_outputs, 
                labeled_labels,
                sdm=sdm  # Tái sử dụng SDM đã tính
            )
            
            # Loss cho nhánh chính: trung bình CE và Dice có trọng số biên
            loss_main = (loss_boundary_ce + loss_boundary_dice) / 2
            
            # --- Bước 3: NHÁNH PHỤ - Standard Loss (ổn định) ---
            # Auxiliary outputs dùng loss tiêu chuẩn để đảm bảo ổn định
            loss_ce_aux1 = ce_loss(outputs_aux1[:args.labeled_bs],
                                   labeled_labels[:].long())
            loss_ce_aux2 = ce_loss(outputs_aux2[:args.labeled_bs],
                                   labeled_labels[:].long())
            loss_ce_aux3 = ce_loss(outputs_aux3[:args.labeled_bs],
                                   labeled_labels[:].long())

            loss_dice_aux1 = dice_loss(
                outputs_aux1_soft[:args.labeled_bs], labeled_labels.unsqueeze(1))
            loss_dice_aux2 = dice_loss(
                outputs_aux2_soft[:args.labeled_bs], labeled_labels.unsqueeze(1))
            loss_dice_aux3 = dice_loss(
                outputs_aux3_soft[:args.labeled_bs], labeled_labels.unsqueeze(1))

            # Loss cho các nhánh phụ: trung bình của 3 auxiliary heads
            supervised_loss_aux = (loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3 +
                                   loss_dice_aux1 + loss_dice_aux2 + loss_dice_aux3) / 6
            
            # --- Logging compatibility: giữ biến cũ để log ---
            loss_ce = loss_boundary_ce  # Log CE của main branch
            loss_dice = loss_boundary_dice  # Log Dice của main branch
            loss_boundary = loss_main  # Để log boundary loss
            # =================================================================

            # ============ CONSISTENCY LOSS (giữ nguyên từ baseline) ============
            preds = (outputs_soft+outputs_aux1_soft +
                     outputs_aux2_soft+outputs_aux3_soft)/4

            variance_main = torch.sum(kl_distance(
                torch.log(outputs_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_main = torch.exp(-variance_main)

            variance_aux1 = torch.sum(kl_distance(
                torch.log(outputs_aux1_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux1 = torch.exp(-variance_aux1)

            variance_aux2 = torch.sum(kl_distance(
                torch.log(outputs_aux2_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(kl_distance(
                torch.log(outputs_aux3_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist_main = (
                preds[args.labeled_bs:] - outputs_soft[args.labeled_bs:]) ** 2

            consistency_loss_main = torch.mean(
                consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(variance_main)

            consistency_dist_aux1 = (
                preds[args.labeled_bs:] - outputs_aux1_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux1 = torch.mean(
                consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

            consistency_dist_aux2 = (
                preds[args.labeled_bs:] - outputs_aux2_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux2 = torch.mean(
                consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

            consistency_dist_aux3 = (
                preds[args.labeled_bs:] - outputs_aux3_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux3 = torch.mean(
                consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

            consistency_loss = (consistency_loss_main + consistency_loss_aux1 +
                                consistency_loss_aux2 + consistency_loss_aux3) / 4
            # ===================================================================
            
            # ============ TOTAL LOSS - CÔNG THỨC MỚI ============
            # loss = loss_main (boundary-aware) + loss_aux (standard) + consistency
            # Trong đó:
            # - loss_main: (boundary_ce + boundary_dice) / 2 cho nhánh chính
            # - supervised_loss_aux: standard loss cho các nhánh phụ
            # - consistency_loss: unsupervised consistency regularization
            loss = loss_main + supervised_loss_aux + consistency_weight * consistency_loss
            # ====================================================
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            
            # ============ LOGGING (THÊM BOUNDARY LOSS) ============
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            # Log Boundary-Aware Loss (NEW)
            writer.add_scalar('info/boundary_loss', loss_boundary, iter_num)
            writer.add_scalar('info/boundary_ce_loss', loss_boundary_ce, iter_num)
            writer.add_scalar('info/boundary_dice_loss', loss_boundary_dice, iter_num)
            # ======================================================
            
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, boundary_loss: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_boundary.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_ds(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    
    # ============ SAVE FINAL MODEL (đảm bảo luôn có model để test) ============
    # Nếu chưa có best_model (do chưa đạt validation checkpoint), save model cuối
    final_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    if not os.path.exists(final_best_path):
        torch.save(model.state_dict(), final_best_path)
        logging.info("No best model found, saved final model to {}".format(final_best_path))
    # Luôn save checkpoint cuối cùng
    final_checkpoint = os.path.join(snapshot_path, 'iter_{}_final.pth'.format(iter_num))
    torch.save(model.state_dict(), final_checkpoint)
    logging.info("Saved final checkpoint to {}".format(final_checkpoint))
    # =========================================================================
    
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("=" * 50)
    logging.info("URPC with Boundary-Aware Loss (SDM Integration)")
    logging.info("Author: KhangPX")
    logging.info("=" * 50)
    train(args, snapshot_path)
