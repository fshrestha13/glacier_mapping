#!/usr/bin/env python
"""
Test Pipeline:
    1- Initialize test data
    2- Initialize the framework
    4- Saves an output csv file in the specified directory
    5- Save images in output directory
"""
from pathlib import Path
from src.models.unet import Unet
import glob
import numpy as np
from matplotlib import pyplot as plt
import torch
import src.utils.metrics as m
import pandas as pd
import argparse
import os
import yaml
from addict import Dict

# def unnormalize(x, conf=1, channels=(0,2,1)):
#     conf = "./data/glaciers_hkh/processed/stats.json"
#     j = json.load(open(conf))
    
#     mean = j['means']
#     std = j['stds']

#     for i,channel in enumerate(channels):
#         x[:,:,i] = x[:,:,i]*std[channel]
#         x[:,:,i] += mean[channel]

#     return x/255
    
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

def save_image(savepath, x, y, y_hat):
    savepath = "."+savepath.split(".")[1]+".jpg"
    x = np.transpose(np.squeeze(x), (1,2,0))

    plt.subplot(141)
    plt.title("Original image")
    _x = unnormalize(x[:,:,[0,2,1]], channels=(0,2,1))
    plt.imshow(_x)

    plt.subplot(142)
    plt.title("542 image")
    _x = unnormalize(x[:,:,[4,3,1]], channels=(4,3,1))
    plt.imshow(_x)

    plt.subplot(143)
    plt.title("True Label")
    plt.imshow(np.squeeze(y))

    plt.subplot(144)
    plt.title("Generated Mask")
    plt.imshow(y_hat)
    
    plt.savefig(savepath)

def get_args():
    parser = argparse.ArgumentParser(
        description='Get test metrics on trained models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-p', '--path', type=str, 
        default=os.environ["DATA_DIR"], 
        help='Base path', 
        dest='basepath'
    )
    parser.add_argument(
        '-s', '--numsamples', type=int, 
        default=-1, 
        help='Number of random samples to use from the test set (Default 20)', 
        dest='numsamples'
    )
    conf_dir = Path(os.environ["ROOT_DIR"], "conf")
    parser.add_argument(
        '-t', '--trainconf', type=str, 
        default=str(conf_dir / "train.yaml"),
        help='Configuration File for training', 
        dest='trainconf'
    )
    parser.add_argument(
        '-c', '--testconf', type=str, 
        default=str(conf_dir / "test.yaml"),
        help='Configuration File for testing', 
        dest='testconf'
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    trainconf = Dict(yaml.safe_load(open(args.trainconf, "r")))
    testconf = Dict(yaml.safe_load(open(args.testconf, "r")))
    testpath = args.basepath+"/processed/test/*"

    sat_imgs = sorted(glob.glob(testpath))
    fname, X, y = get_x_y(sat_imgs,args.numsamples)
    
    for run_name in testconf.run_names:
        # Iterate over different runs
        for model in testconf.models:
        # Iterate over different models
            model_path = f"{args.basepath}/models/{run_name}/{model}.pt"
            if not os.path.exists(model_path):
                print(f"{model_path} does not exist")
                break      
            # Load the U-Net model
            state_dict = torch.load(model_path, map_location="cpu")
            model = Unet(trainconf.model_opts.args.inchannels, 
                                trainconf.model_opts.args.outchannels, 
                                trainconf.model_opts.args.net_depth, 
                                trainconf.model_opts.args.channel_layer)    
            model.load_state_dict(state_dict())
            metric_results = {
                "model_path" : [],
                "fname" : [],
                "pred_fname" : [],
                "pixel_acc" : [],
                "precision" : [],
                "recall" : [],
                "IoU" : []
            }
            savepath = f"{args.basepath}/processed/preds/{run_name}/{model}"
            if not os.path.exists(savepath):
                os.makedirs(savepath)
                
            for _name, _x, _y in zip(fname, X, y):
                y_fname = _name.replace("img","mask")
                y = np.load(y_fname)
                # get unet prediction
                with torch.no_grad():
                    _x = _x.transpose((2,0,1))
                    _x = np.expand_dims(_x, axis=0)
                    y_hat = model(torch.from_numpy(_x))
                    y_hat = torch.sigmoid(y_hat)
                    y_hat = torch.squeeze(y_hat).numpy()
                    save_image(savepath+_name.split("/")[-1],_x,y,y_hat)
                _y = torch.from_numpy(np.array(_y, dtype=bool))

                metric_results["model_path"].append(model_path)
                metric_results["fname"].append(_name)
                metric_results["pred_fname"].append(savepath+_name.split("/")[-1])
                metric_results["pixel_acc"].append(m.pixel_acc(y_hat > testconf.threshold.pixel_acc,_y))
                metric_results["precision"].append(m.precision(y_hat > testconf.threshold.precision,_y))
                metric_results["recall"].append(m.recall(y_hat > testconf.threshold.recall,_y))
                metric_results["IoU"].append(m.IoU(y_hat > testconf.threshold.IoU,_y))

                pd.DataFrame(metric_results).to_csv(savepath+"/results.csv", index=False)

    # for model in models:
    #     savepath = basepath+"/processed/preds/"+model.split(".")[0]+"/"
    #     if not os.path.exists(savepath):
    #         os.makedirs(savepath)
    #     gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
    #     for key in gen:
    #         vars()["model_"+key] = []

    #     model_path = basepath+"/models/no_sigmoid/"+model
    #     # Load the U-Net model
    #     state_dict = torch.load(model_path, map_location="cpu")
    #     model = UnetDropout(12, 1, 3)
    #     model.load_state_dict(state_dict())

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
    #             # y_hat = torch.from_numpy(_y_hat > threshold) 
    #             save_image(savepath+_name.split("/")[-1],_x,y,y_hat)
    #         _y = torch.from_numpy(np.array(_y, dtype=bool))

    #         metric_results["fname"].append(_name)
    #         metric_results["model_path"].append(model_path)
    #         metric_results["pred_fname"].append(savepath+_name.split("/")[-1])
    #         # get channel wise metrices and compute mean

    #         gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
    #         for key in gen:
    #             vars()[key] = []

    #         for k in range(_y.shape[2]):
    #             mean_precision.append(m.precision(_y[:, :, k], y_hat[:, :, k]))
    #             tp,fp,fn = m.tp_fp_fn(_y[:, :, k], y_hat[:, :, k])
    #             mean_tp.append(tp)
    #             mean_fp.append(fp)
    #             mean_fn.append(fn)
    #             mean_pixel_acc.append(m.pixel_acc(_y[:, :, k], y_hat[:, :, k]))
    #             mean_dice.append(m.dice(_y[:, :, k], y_hat[:, :, k]))
    #             mean_recall.append(m.recall(_y[:, :, k], y_hat[:, :, k]))
    #             mean_IoU.append(m.IoU(_y[:, :, k], y_hat[:, :, k]))

    #         gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
    #         for key in gen:
    #             _ = vars()[key]
    #             metric_results[key].append(mean(_))

    #         gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
    #         for key in gen:
    #             model_list = vars()["model_"+key]
    #             k = vars()[key]
    #             model_list.append(mean(k))

    #     # Append to mean_metric_results
    #     mean_metric_results["model_path"] = model_path
    #     mean_metric_results["fname"] = "Test set"
    #     mean_metric_results["pred_fname"] = savepath
    #     gen = (x for x in metric_results if x not in ["model_path","fname","pred_fname"])
    #     for key in gen:
    #         model_list = vars()["model_"+key]
    #         mean_metric_results[key].append(mean(model_list))
    
    # # print(metric_results)
    # pd.DataFrame(mean_metric_results).to_csv(savepath+"/mean_results.csv", index=False)
    # pd.DataFrame(metric_results).to_csv(savepath+"/results.csv", index=False)