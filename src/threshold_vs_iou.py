#!/usr/bin/env python
# Use this script to calculate the threshold that gives best iou for a particular trained model
# Usage: python3 -m src.threshold_vs_iou -n [run_name] -m [model_name]
# Example: python3 -m src.threshold_vs_iou -n test_16 -m model_5
from pathlib import Path
from src.models.unet_dropout import UnetDropout
from src.models.unet import Unet
import argparse
import glob
import numpy as np
from matplotlib import pyplot as plt
import torch
import os
import yaml
from addict import Dict
import src.metrics as m
    
def get_x_y(sat_imgs, n):
    list_x = []
    list_y = []
    list_imgpath = []
    path = "/".join(sat_imgs[0].split("/")[:-1])
    for fname in sat_imgs:
        n_slice = fname.split("/")[-1].split("_")[1]
        n_img = fname.split("/")[-1].split("_")[3].split(".")[0]
        imagepath = f"{path}/slice_{n_slice}_img_{n_img}.npy"
        maskpath = f"{path}/slice_{n_slice}_mask_{n_img}.npy"
        x = np.load(imagepath)
        y = np.load(maskpath)
        y = np.expand_dims(y, axis=2)
        list_x.append(x)
        list_y.append(y)
        list_imgpath.append(imagepath)
    return list_imgpath[:n], np.asarray(list_x[:n]), np.asarray(list_y[:n])

def get_args():
    parser = argparse.ArgumentParser(description='Get IoU vs threshold on trained models',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--path', type=str, default='./data/glaciers/', help='Base path', dest='basepath')
    parser.add_argument('-m', '--modelname', type=str, default="model_150", help='Saved model name', dest='model')
    parser.add_argument('-n', '--name', type=str, help='Name of run', dest='run_name', required=True)
    parser.add_argument('-s', '--numsamples', type=int, default=20, help='Number of random samples to use from the test set (Default 20)', dest='numsamples')
    parser.add_argument('-min', '--min', type=int, default=0, help='Minimum threshold value (Default 0)', dest='min')
    parser.add_argument('-max', '--max', type=int, default=1.05, help='Maximum threshold value (Default 1.05)', dest='max')
    parser.add_argument('-interval', '--interval', type=int, default=0.05, help='Interval for threshold (Default 0.05)', dest='interval')
    parser.add_argument('-c', '--conf', type=str, default='./conf/train_conf.yaml', help='Configuration File for training', dest='conf')

    return parser.parse_args()

if __name__ == '__main__':
    # Define paths and threshold
    args = get_args()
    conf = Dict(yaml.safe_load(open(args.conf, "r")))
    testpath = args.basepath+"/processed/test/*"
    # Get X and y
    sat_imgs = sorted(glob.glob(testpath))
    fname, X, y = get_x_y(sat_imgs,args.numsamples)
    # Set initial vectors
    thresholds = np.arange(args.min,args.max,args.interval)
    ious = np.zeros(len(thresholds))
    model_path = f"{args.basepath}models/{args.run_name}/{args.model}.pt"
    # Load the U-Net model
    state_dict = torch.load(model_path, map_location="cpu")
    if conf.model_opts.name == "UnetDropout":
        model = UnetDropout(conf.model_opts.args.inchannels, 
                            conf.model_opts.args.outchannels, 
                            conf.model_opts.args.net_depth, 
                            channel_layer = conf.model_opts.args.channel_layer)
    elif conf.model_opts.name == "Unet":
        model = Unet(conf.model_opts.args.inchannels, 
                            conf.model_opts.args.outchannels, 
                            conf.model_opts.args.net_depth, 
                            conf.model_opts.args.channel_layer)    
    model.load_state_dict(state_dict())
    
    savepath = f"{args.basepath}/processed/iou/{args.run_name}/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    for i, threshold in enumerate(thresholds):
        mean_iou = []
        for _name, _x, _y in zip(fname, X, y):
            y_fname = _name.replace("img","mask")
            y = np.load(y_fname)
            _x = np.expand_dims(_x, axis=0)
            _y = np.expand_dims(y, axis=0)
            # get unet prediction
            with torch.no_grad():
                _x = _x.transpose((0,3,1,2))
                y_hat = model(torch.from_numpy(_x))
                _y_hat = torch.sigmoid(y_hat).numpy()
                y_hat = torch.from_numpy(_y_hat > threshold) 
            _y = torch.from_numpy(np.array(_y, dtype=bool))
            mean_iou.append(m.IoU(torch.squeeze(y_hat), torch.squeeze(_y)))
        ious[i] = np.mean(mean_iou)
        print(f"Threshold: {threshold}, IoU: {ious[i]}")

    plt.ylim((0,0.5))
    plt.xlabel("Threshold")
    plt.ylabel("IoU")
    plt.plot(thresholds, ious)
    plt.savefig(f"{savepath}/{args.model}.jpg")

#     for _name, _x, _y in zip(fname, X, y):
#         y_fname = _name.replace("img","mask")
#         y = np.load(y_fname)
#         # get unet prediction
#         with torch.no_grad():
#             _x = _x.transpose((2,0,1))
#             _x = np.expand_dims(_x, axis=0)
#             y_hat = model(torch.from_numpy(_x))
#             y_hat = torch.sigmoid(y_hat)
#             y_hat = torch.squeeze(y_hat).numpy()
#             # print(y_hat)
#             y_hat = torch.from_numpy(y_hat > 0.35) 
#             save_image(savepath+_name.split("/")[-1],_x,y,y_hat)
# #         _y = torch.from_numpy(np.array(_y, dtype=bool))

# #         metric_results["fname"].append(_name)
# #         metric_results["model_path"].append(model_path)
# #         metric_results["pred_fname"].append(savepath+_name.split("/")[-1])
# #         # get channel wise metrices and compute mean

# #         gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
# #         for key in gen:
# #             vars()[key] = []

# #         for k in range(_y.shape[2]):
# #             mean_precision.append(m.precision(_y[:, :, k], y_hat[:, :, k]))
# #             tp,fp,fn = m.tp_fp_fn(_y[:, :, k], y_hat[:, :, k])
# #             mean_tp.append(tp)
# #             mean_fp.append(fp)
# #             mean_fn.append(fn)
# #             mean_pixel_acc.append(m.pixel_acc(_y[:, :, k], y_hat[:, :, k]))
# #             mean_dice.append(m.dice(_y[:, :, k], y_hat[:, :, k]))
# #             mean_recall.append(m.recall(_y[:, :, k], y_hat[:, :, k]))
# #             mean_IoU.append(m.IoU(_y[:, :, k], y_hat[:, :, k]))

# #         gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
# #         for key in gen:
# #             _ = vars()[key]
# #             metric_results[key].append(mean(_))

# #         gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
# #         for key in gen:
# #             model_list = vars()["model_"+key]
# #             k = vars()[key]
# #             model_list.append(mean(k))

# #     # Append to mean_metric_results
# #     mean_metric_results["model_path"] = model_path
# #     mean_metric_results["fname"] = "Test set"
# #     mean_metric_results["pred_fname"] = savepath
# #     gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
# #     for key in gen:
# #         model_list = vars()["model_"+key]
# #         mean_metric_results[key].append(mean(model_list))

# # # print(metric_results)
# # pd.DataFrame(mean_metric_results).to_csv(savepath+"/mean_results.csv", index=False)
# # pd.DataFrame(metric_results).to_csv(savepath+"/results.csv", index=False)