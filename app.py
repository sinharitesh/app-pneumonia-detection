import argparse
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import gradio as gr
import tqdm
from fnutils import *


label_folder   =   './labels/'
yolo_folder    =   "../yolov5/"
model_path     =   "./models/yolo-exp2-best.pt"
target_dir     =   "/tmp/"
image_dir      =   './images/' # For debug mode




def parse_args():
    parser = argparse.ArgumentParser(description='Object detection app parser')
    parser.add_argument('--mode', type=str, default='predict', help='can be used in predict and predict_debug modes')
    args = parser.parse_args()
    return(args)

if __name__ == "__main__":
    args = parse_args()
    if (args.mode == "predict"):
        outputs = [gr.outputs.Image(label = "Prediction", type = "file")]
        iface = gr.Interface(
            image_predict, 
            gr.inputs.File(type = "file", keep_filename = True), 
            outputs,
            capture_session=True)

        iface.launch(share = True)
    elif (args.mode == "predict_debug"):
        outputs = [gr.outputs.Image(label = "Ground Truth")
               , gr.outputs.Image(label = "Prediction")]
        iface = gr.Interface(
            image_predict_and_debug, 
            gr.inputs.File(type = "file", keep_filename = True), 
            outputs,
            capture_session=True)

        iface.launch(share = True)
    else:
        print("NO MODE DETECTED, QUITTING")