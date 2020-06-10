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
    
def get_x_y(sat_imgs, n):
    list_x = []
    list_y = []
    list_imgpath = []
    path = "/".join(sat_imgs[0].split("/")[:-1])
    for fname in sat_imgs:
        if "img" in fname:
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
    savepath = savepath.split(".")[0]+".jpg"
    x = np.transpose(np.squeeze(x), (1,2,0))

    plt.subplot(141)
    plt.title("Normalized image")
    plt.imshow(x[:,:,[2,1,0]])

    plt.subplot(142)
    plt.title("542 image")
    plt.imshow(x[:,:,[5,4,2]])

    plt.subplot(143)
    plt.title("True Label")
    plt.imshow(np.squeeze(y))

    plt.subplot(144)
    plt.title("Generated Mask")
    plt.imshow(y_hat)
    
    plt.savefig(savepath)

def get_results(y_hat, y, threshold):
    y = torch.from_numpy(y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.squeeze(y)
    metric_results = {
        "pixel_acc" : [],
        "precision" : [],
        "recall" : [],
        "IoU" : []
    }
    for k, v in metric_results.items():
        if k in threshold.keys():
            _y_hat = y_hat > threshold[k]
            y = y.bool()
        metric_fun = getattr(m, k)
        metric_value = metric_fun(_y_hat.to(device), y.to(device))
        metric_results[k] = metric_value
    return metric_results


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
        model_results = {
                "model_path" : [],
                "pixel_acc" : [],
                "precision" : [],
                "recall" : [],
                "IoU" : []
        }
        for model_name in testconf.models:
        # Iterate over different models
            model_path = f"{args.basepath}/models/{run_name}/{model_name}.pt"
            if not os.path.exists(model_path):
                print(f"{model_path} does not exist")
                break      
            # Load the U-Net model
            state_dict = torch.load(model_path, map_location="cpu")
            model = Unet(trainconf.model_opts.args.inchannels, 
                                trainconf.model_opts.args.outchannels, 
                                trainconf.model_opts.args.net_depth, 
                                trainconf.model_opts.args.channel_layer)    
            # Might wanna change this
            model.load_state_dict(state_dict)
            results = {
                "model_path" : [],
                "fname" : [],
                "pred_fname" : [],
                "pixel_acc" : [],
                "precision" : [],
                "recall" : [],
                "IoU" : []
            }
            savepath = f"{args.basepath}/processed/preds/{run_name}/{model_name}/"
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
                    y_hat = torch.squeeze(y_hat)
                    save_image(savepath+_name.split("/")[-1],_x,y,y_hat)
                results["model_path"].append(model_path)
                results["fname"].append(_name)
                results["pred_fname"].append(savepath+_name.split("/")[-1])
                metric_results = get_results(y_hat,y,testconf.threshold)
                for k, v in metric_results.items():
                    results[k].append(v)
            pd.DataFrame(results).to_csv(savepath+"/results.csv", index=False)
        model_results["model_path"].append(model_path)
        for k, v in metric_results.items():
            model_results[k].append(np.mean(results[k]))
        pd.DataFrame(model_results).to_csv(savepath+"/mean_results.csv", index=False)