import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
# New: Add model_path argument for Kaggle compatibility
parser.add_argument('--model_path', type=str, default=None,
                    help='Path to model directory. If None, uses ../model/{exp}_{labeled_num}_labeled/')
parser.add_argument('--save_prediction', type=int, default=1,
                    help='whether to save prediction results (1=yes, 0=no)')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    
    # Use model_path if provided, otherwise use default structure
    if FLAGS.model_path is not None:
        snapshot_path = FLAGS.model_path
    else:
        snapshot_path = "../model/{}_{}_labeled".format(
            FLAGS.exp, FLAGS.labeled_num)
    
    test_save_path = os.path.join(snapshot_path, "{}_predictions/".format(FLAGS.model))
    
    print(f"="*50)
    print(f"Model path: {snapshot_path}")
    print(f"Predictions will be saved to: {test_save_path}")
    print(f"="*50)
    
    if FLAGS.save_prediction:
        if os.path.exists(test_save_path):
            shutil.rmtree(test_save_path)
        os.makedirs(test_save_path, exist_ok=True)
    
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    
    # Try to load best model
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    
    if not os.path.exists(save_mode_path):
        # Fallback to final_model.pth
        save_mode_path = os.path.join(snapshot_path, 'final_model.pth')
    
    if not os.path.exists(save_mode_path):
        raise FileNotFoundError(f"Model not found at {save_mode_path}")
    
    # Load checkpoint (handle both old and new checkpoint formats)
    checkpoint = torch.load(save_mode_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint (iter: {checkpoint.get('iteration', 'N/A')}, dice: {checkpoint.get('best_performance', 'N/A')})")
    else:
        net.load_state_dict(checkpoint)
    
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric, snapshot_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, model_path = Inference(FLAGS)
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Model: {model_path}")
    print(f"Class 1 (RV):  Dice={metric[0][0]:.4f}, HD95={metric[0][1]:.4f}, ASD={metric[0][2]:.4f}")
    print(f"Class 2 (Myo): Dice={metric[1][0]:.4f}, HD95={metric[1][1]:.4f}, ASD={metric[1][2]:.4f}")
    print(f"Class 3 (LV):  Dice={metric[2][0]:.4f}, HD95={metric[2][1]:.4f}, ASD={metric[2][2]:.4f}")
    print("-"*50)
    avg = (metric[0]+metric[1]+metric[2])/3
    print(f"Average:       Dice={avg[0]:.4f}, HD95={avg[1]:.4f}, ASD={avg[2]:.4f}")
    print("="*50)
