import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import tqdm
import random
import string


label_folder   =   './labels/'
yolo_folder    =   "../yolov5/"
model_path     =   "./models/yolo-exp2-best.pt"
target_dir     =   "/tmp/"
image_dir      =   './images/' # For debug mode


model = torch.hub.load( yolo_folder, 'custom', path= model_path, source='local')  # local repo

def get_bboxes_xxyy(str_label):
    l_master = [0,0,1,1]
    try:
        elements = str_label.split(" "); num_labels = len(elements)/6; #print(int(num_labels))
        l_master = []
        for i in range(int(num_labels)):
            l = []
            l.append(elements[i * 6 + 2])
            l.append(elements[i * 6 + 3])
            l.append(elements[i * 6 + 4])
            l.append(elements[i * 6 + 5])
            l = [ float(x) for x in l ]
            l_master.append(l)
    except Exception as e:
            pass
    return(l_master)

def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], 
                         fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)
    
def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt, verticalalignment='top', color='white',
                   fontsize=sz, weight='bold')
    draw_outline(text, 1)
    
def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
    
def show_img(im, figsize=None, ax=None, cmap = None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax
    
def save_image_bb(im, ax, bbs, lbl, save_file = "default.png"):
    ax = show_img(im = im, ax = ax, cmap=plt.get_cmap('gray'))
    for i, bx in enumerate(bbs):
        b = bb_wh_from_xxyy(bx)
        draw_rect(ax, b)
        draw_text(ax, b[:2], lbl)
    plt.savefig(save_file)
    plt.close()
    
def bb_wh_from_xxyy(a): return np.array([a[0],a[1],a[2]-a[0], a[3]-a[1]])

def  read_bbstr(f ):
    ret = ""
    #f = 'badf2d31cdbd.png'
    #folder = './gradio-siim-app/test/labels/'
    label_folder   = './labels/'
    with open(label_folder + f.split('.')[0] + "-boundingbox.txt", 'r') as fl:
        ret = (fl.readline())
    return(ret)

def  read_label(f ):
    ret = ""
    label_folder   = './labels/'
    with open(label_folder + f.split('.')[0] + "-label.txt", 'r') as fl:
        ret = (fl.readline())
    return(ret)



def get_pred_bb_label_from_df(df_results):
    bbs = []
    lbls = []
    probs = []
    for i in range(df_results.shape[0]):
        bbxyxy = []
        xmin = df_results['xmin'][i]
        ymin = df_results['ymin'][i]
        xmax = df_results['xmax'][i]
        ymax = df_results['ymax'][i]
        bbxyxy.append(xmin)
        bbxyxy.append(ymin)
        bbxyxy.append(xmax)
        bbxyxy.append(ymax)
        label = df_results['name'][i]
        prob = round(float(df_results['confidence'][i]),2)
        bbs.append(bbxyxy)
        lbls.append(label)
        probs.append(prob)
    return(bbs, lbls, probs)

def get_orig_file_name(fname):
# This function is required for gradio uploads.
# When a file is loaded , the name is changed.
# this function reqturns the orig file name which
# is useful in predict and debug scenario.
    fname = fname.replace("/tmp/", "")
    ext = fname.split(".")[-1]
    temp = fname.split(".")[-2]
    fl = temp.split("_")[0]
    ret = fl + "." + ext
    return(ret)

def save_original_img_with_bb(f_with_ext
                              , target_dir
                               , image_dir 
                             ):
    f = f_with_ext
    f_without_ext = f.split('.')[0]
    bbstr = read_bbstr(f.split('.')[0])
    lbl = read_label(f.split('.')[0])
    bboxes = get_bboxes_xxyy(bbstr)
    im = Image.open(image_dir + f)
    file_target = f"{target_dir}{f_without_ext}-orig.png"
    figsize = (10,10)
    _, ax = plt.subplots(figsize = figsize)
    save_image_bb(im, ax , bboxes, lbl, save_file = file_target )
    return(file_target)

def get_prediction_df(image_file):
    df_results = pd.DataFrame()
    file_results = model(image_file)
    for i in range(len(file_results)):
        df_temp = file_results.pandas().xyxy[i]
        df_temp['filename'] = file_results.files[i]
        if (df_temp.shape[0] > 0):
            df_results = df_results.append(df_temp)
    return(df_results)


def save_predicted_img_with_bb_new( f_with_ext, uploaded_file
                              , target_dir 
                               , image_dir ):
    f = f_with_ext
    f_without_ext = f.split('.')[0]
    df_results = get_prediction_df(uploaded_file)
    bboxes, labels, probs = get_pred_bb_label_from_df(df_results)
    # creating output string #
    str_output = ""
    try:
        lblstr = (' ').join(labels)
        strprobs = [str(prob) for prob in probs ]
        probstr = (' ').join((strprobs))
        str_output = lblstr + " " + probstr;
    except:
        pass
    
    im = Image.open(uploaded_file)
    file_target = f"{target_dir}{f_without_ext}-pred.png"
    lbl = ""
    try:
        lbl = labels[0]
    except:
        pass
    figsize = (10,10)
    _, ax = plt.subplots(figsize = figsize)
    save_image_bb(im, ax , bboxes, lbl, save_file = file_target )
    return(file_target, str_output)


def generate_temp_filename() -> str:
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    return(random_string)


def image_predict_new(im):
    img = Image.fromarray(im)
    tmp_file_name = generate_temp_filename() + ".png"
    tmp_file_path = f"/tmp/{tmp_file_name}"
    img.save(tmp_file_path)
    pred_file_name, labelstr = save_predicted_img_with_bb_new( tmp_file_name , tmp_file_path ,  target_dir, image_dir)
    print('pred_file_name', pred_file_name, tmp_file_path, labelstr)
    imret = Image.open(pred_file_name)
    return(imret, labelstr)


def image_predict_and_debug(file):
    im = Image.open(file)
    orig_file = get_orig_file_name(file.name)
    orig_file_name = save_original_img_with_bb(orig_file, target_dir, image_dir)
    pred_file_name, labelstr = save_predicted_img_with_bb_new(orig_file , file.name ,  target_dir, image_dir)
    return(orig_file_name, pred_file_name)

def image_predict(file):
    pred_file_name = save_predicted_img_with_bb_new(orig_file , file.name ,  target_dir, image_dir)
    return(pred_file_name)


