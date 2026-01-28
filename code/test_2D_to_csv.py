"""
Test script với xuất kết quả ra CSV
Author: KhangPX
"""
import argparse
import os
import shutil
import csv
from datetime import datetime

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/URPC_Boundary_Aware_v2', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_urpc', help='model_name')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--run_id', type=int, default=1,
                    help='Run ID for experiment tracking')
parser.add_argument('--csv_output', type=str, default='',
                    help='Path to save CSV results')
parser.add_argument('--model_path', type=str, default='',
                    help='Custom path to model weights (optional)')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0, 100.0, 100.0
    dice = metric.binary.dc(pred, gt)
    try:
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
    except:
        asd = 100.0
        hd95 = 100.0
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
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds" or FLAGS.model == "unet_urpc":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    if test_save_path:
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
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions_run{}/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model, FLAGS.run_id)
    
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    
    # Load model weights
    if FLAGS.model_path and os.path.exists(FLAGS.model_path):
        save_mode_path = FLAGS.model_path
    else:
        save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    
    net.load_state_dict(torch.load(save_mode_path))
    print("Loaded weights from: {}".format(save_mode_path))
    net.eval()

    # Collect metrics per case
    all_metrics = []
    first_total = np.zeros(3)
    second_total = np.zeros(3)
    third_total = np.zeros(3)
    
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        
        all_metrics.append({
            'case': case,
            'RV_dice': first_metric[0], 'RV_hd95': first_metric[1], 'RV_asd': first_metric[2],
            'Myo_dice': second_metric[0], 'Myo_hd95': second_metric[1], 'Myo_asd': second_metric[2],
            'LV_dice': third_metric[0], 'LV_hd95': third_metric[1], 'LV_asd': third_metric[2],
        })
    
    # Calculate averages
    num_cases = len(image_list)
    avg_metrics = {
        'run_id': FLAGS.run_id,
        'exp': FLAGS.exp,
        'model': FLAGS.model,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'RV_dice': first_total[0] / num_cases,
        'RV_hd95': first_total[1] / num_cases,
        'RV_asd': first_total[2] / num_cases,
        'Myo_dice': second_total[0] / num_cases,
        'Myo_hd95': second_total[1] / num_cases,
        'Myo_asd': second_total[2] / num_cases,
        'LV_dice': third_total[0] / num_cases,
        'LV_hd95': third_total[1] / num_cases,
        'LV_asd': third_total[2] / num_cases,
        'mean_dice': (first_total[0] + second_total[0] + third_total[0]) / (3 * num_cases),
        'mean_hd95': (first_total[1] + second_total[1] + third_total[1]) / (3 * num_cases),
        'mean_asd': (first_total[2] + second_total[2] + third_total[2]) / (3 * num_cases),
    }
    
    # Save to CSV if specified
    if FLAGS.csv_output:
        csv_dir = os.path.dirname(FLAGS.csv_output)
        if csv_dir and not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        
        file_exists = os.path.exists(FLAGS.csv_output)
        with open(FLAGS.csv_output, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=avg_metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(avg_metrics)
        print(f"Results saved to: {FLAGS.csv_output}")
    
    return avg_metrics


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    results = Inference(FLAGS)
    
    print("\n" + "="*60)
    print(f"Run {FLAGS.run_id} Results:")
    print("="*60)
    print(f"RV  - Dice: {results['RV_dice']:.4f}, HD95: {results['RV_hd95']:.2f}, ASD: {results['RV_asd']:.2f}")
    print(f"Myo - Dice: {results['Myo_dice']:.4f}, HD95: {results['Myo_hd95']:.2f}, ASD: {results['Myo_asd']:.2f}")
    print(f"LV  - Dice: {results['LV_dice']:.4f}, HD95: {results['LV_hd95']:.2f}, ASD: {results['LV_asd']:.2f}")
    print("-"*60)
    print(f"MEAN - Dice: {results['mean_dice']:.4f}, HD95: {results['mean_hd95']:.2f}, ASD: {results['mean_asd']:.2f}")
    print("="*60)
