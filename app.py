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
import random
import string
from fnutils import *


label_folder   =   './labels/'
yolo_folder    =   "../yolov5/"
model_path     =   "./models/yolo-exp2-best.pt"
target_dir     =   "/tmp/"
image_dir      =   './images/' # For debug mode

# def save_predicted_img_with_bb( f_with_ext, uploaded_file
#                               , target_dir 
#                                , image_dir ):
#     f = f_with_ext
#     f_without_ext = f.split('.')[0]
#     df_results = get_prediction_df(uploaded_file)
#     bboxes, labels = get_pred_bb_label_from_df(df_results)
#     im = Image.open(uploaded_file)
#     file_target = f"{target_dir}{f_without_ext}-pred.png"
#     lbl = ""
#     try:
#         lbl = labels[0]
#     except:
#         pass
#     figsize = (10,10)
#     _, ax = plt.subplots(figsize = figsize)
#     save_image_bb(im, ax , bboxes, lbl, save_file = file_target )
#     #if len(bboxes) > 1: print("P", len(bboxes), f_with_ext)
#     return(file_target)


def parse_args():
    parser = argparse.ArgumentParser(description='Object detection app parser')
    parser.add_argument('--mode', type=str, default='predict', help='can be used in predict and predict_debug modes')
    args = parser.parse_args()
    return(args)


if __name__ == "__main__":
    args = parse_args()
    if (args.mode == "predict"):
        outputs = [gr.outputs.Image(label = "Prediction"), gr.outputs.Label(label = "label|confidence")]
        iface = gr.Interface(
            image_predict_new, 
            gr.inputs.Image(label = "input_image"), 
            outputs,
            flagging_options = ["bounding boxes missed","label incorrect"],
            examples = [["images/28e9e044dc31.png"]  ,["images/6fb675554ba4.png"]],
            capture_session=True)

        iface.launch(share = True)
    elif (args.mode == "debug"):
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