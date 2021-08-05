#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.nn.functional as F
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import patches, patheffects
import shutil


# In[3]:


lst_categories = []
lst_categories.append({'id': 0, 'name' : 'typical'})
lst_categories.append({'id': 1, 'name' : 'indeterminate'})
lst_categories.append({'id': 2, 'name' : 'atypical'})
lst_categories


# In[15]:


def get_category_by_name(categories = lst_categories , name = "none"):
    ele_id = -1
    try:
        ele = next(item for item in categories if item['name'] == name)
        ele_id = ele['id']
    except:
        pass
    return(ele_id)

def get_cat_id(dcmfilename, categories = lst_categories):
    df_ret = df_merge[df_merge['dcm']== dcmfilename].reset_index()
    label = "none"
    if (df_ret.shape[0] == 1):
        if (df_ret['negative'].values == 1):
            label = "none"
        if (df_ret['indeterminate'].values == 1):
            label = "indeterminate"
        if (df_ret['atypical'].values == 1):
            label = "atypical"
        if (df_ret['typical'].values == 1):
            label = "typical"
    cat_id = get_category_by_name(categories, label)
    return(cat_id)

def get_cat_name(dcmfilename, categories = lst_categories):
    df_ret = df_merge[df_merge['dcm']== dcmfilename].reset_index()
    #df_ret['negative'].values
    label = "none"
    if (df_ret.shape[0] == 1):
        if (df_ret['negative'].values == 1):
            label = "none"
        if (df_ret['indeterminate'].values == 1):
            label = "indeterminate"
        if (df_ret['atypical'].values == 1):
            label = "atypical"
        if (df_ret['typical'].values == 1):
            label = "typical"
    return(label)



# In[16]:


df_merge[df_merge['dcm']== '65761e66de9f']


# In[22]:


def get_bbstr(pngfilename):
    df_ret = df_merge[df_merge['png'] == pngfilename].reset_index()
    bbstr = ""
    if (df_ret.shape[0] == 1):
        bbstr = df_ret['label'][0]
    return(bbstr)

def get_bboxes(str_label):
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


# In[17]:


input_folder = './data-siim/train/'
df_image_level = pd.read_csv(f"{input_folder}train_image_level.csv")
df_study_level = pd.read_csv(f'{input_folder}train_study_level.csv')
# TODO: Following needs to changed with rename columns call which will ensure right substitution. 
df_study_level.columns = ["id", "negative", "typical", "indeterminate", "atypical"]
df_study_level['match_id'] = list(map(lambda x: x.replace("_study", ""), df_study_level['id']))
df_image_level['match_id'] = df_image_level['StudyInstanceUID']
df_merge = pd.merge(df_study_level, df_image_level , on = ["match_id"])
df_merge = df_merge.reset_index()
df_merge['dcm'] = list(map(lambda x: x.replace("_image", ""), df_merge['id_y']))
df_merge['png'] = list(map(lambda x: x.replace("_image", ""), df_merge['id_y']))
df_merge['cat_id'] = list(map(get_cat_id, df_merge['dcm']))
df_merge['cat_name'] = list(map(get_cat_name, df_merge['dcm']))
df_merge.head()


# In[108]:


# with open('./data-siim/train/bbtexts/bb4b1da810f3.txt','r',encoding='utf-8') as f:
#     lines = f.readlines()
#     print(lines)


# In[26]:





# In[25]:


lst_categories


# In[52]:


def show_img_gray(im, figsize=None, ax=None, cmap = None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, cmap= cmap, vmin=0, vmax=255)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
    
def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], 
                         fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)
def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt, verticalalignment='top', color='white',
                   fontsize=sz, weight='bold')
    draw_outline(text, 1)
def bb_hw2(a): return np.array([a[0],a[1],a[2]-a[0], a[3]-a[1]]) # this is good.

def get_predicted_bboxes(masked_image):
    lbl_0 = label(masked_image) 
    props = regionprops(lbl_0)
    propbbs = []
    labels = []
    for prop in props:
            tmp = []
            tmp.append(prop.bbox[1])
            tmp.append(prop.bbox[0])
            tmp.append(prop.bbox[3])
            tmp.append(prop.bbox[2])
            #print(, prop.bbox[0], prob.bbox[3], prob.bbox[2])
            propbbs.append(tmp)
            labels.append(prop.label)
            #print('Found bbox', prop.bbox, prop.label) 
    return(propbbs, labels)

def plot_image_bb(im, bbs, lbl, ax = None, figsize =(6,6)):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax = show_img_gray(im = im, ax = ax, figsize = figsize, cmap=plt.get_cmap('gray'))
    for i, bx in enumerate(bbs):
        b = bb_hw2(bx)
        draw_rect(ax, b)
        draw_text(ax, b[:2], lbl)
        
def plot_image_bb_xywh(im, bbs, lbl, ax = None, figsize =(6,6)):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax = show_img_gray(im = im, ax = ax, figsize = figsize, cmap=plt.get_cmap('gray'))
    for i, bx in enumerate(bbs):
        b = bx
        draw_rect(ax, b)
        draw_text(ax, b[:2], lbl)


# In[44]:


from PIL import Image
import matplotlib.pyplot as plt


# In[45]:


folder = './data-siim/train/images/'


# In[46]:





# In[65]:


def bb_xywh(a): return np.array([.5*(a[0]+a[2]) , .5*(a[1] + a[3]), a[2]-a[0], a[3]-a[1]]) 
# here input is in [x1,y1,x2,y2] format


# In[70]:


def normalize_bb(a, width, height): return np.array([a[0]/width , a[1]/height, a[2]/width, a[3]/height]) 


# In[73]:


def get_yolo_bbs(bbs, width,  height):
    retbbs = []
    for bb in bbs:
        b = bb_xywh(bb)
        b = normalize_bb(b, width, height)
        retbbs.append(b)
    return(retbbs)


# In[75]:





# In[76]:



target_dir = './data-siim/train/bbtexts/'


# In[77]:


imgs[0:5]


# In[ ]:





# In[118]:


def write_yolo_textfile(png_no_ext, p_bbs, lbl_id, target_dir = './data-siim/train/labels/' ):
    lbl_id = get_cat_id(png_no_ext)
    if (lbl_id != -1):
        if len(p_bbs) > 0:
            with open(target_dir + png_no_ext + ".txt", 'w') as the_file:
                #print("writing:", png_no_ext)
                for bb in p_bbs:
                    str_write = f'{lbl_id} {bb[0]} {bb[1]} {bb[2]} {bb[3]}\n'
                    the_file.write(str_write)
        else:
            print("bbs not found for:", png_no_ext)
    else:
        pass
        #print("label id:", lbl_id, "skipping file:", png_no_ext, p_bbs)
    return(lbl_id)


# In[177]:


len(os.listdir('./data-siim/valid/images/'))


# In[163]:


# shutil.rmtree('./data-siim/train/labels/')
# ! mkdir './data-siim/train/labels/'


# In[149]:


# ! mkdir './data-siim/train/labels/'


# In[164]:


# Genarating text files for YOLO
# Expand this
lst_lbl_minus_1 = []
lst_lbl_0 = []
lst_lbl_1 = []
lst_lbl_2 = []
imgs = os.listdir('./data-siim/train/images/');
for png_with_ext in imgs:
    png_no_ext = png_with_ext.split(".")[0]
    image_file = folder + png_no_ext + ".png"
    im = Image.open(image_file)
    width, height = im.size
    bbs = get_bboxes(get_bbstr(png_no_ext))
    retbbs = get_yolo_bbs(bbs, width, height)
    lbl_id = get_cat_id(png_no_ext)
    write_yolo_textfile(png_no_ext, retbbs, lbl_id)
    
    if (lbl_id == 0):
        lst_lbl_0.append(png_no_ext)
    if (lbl_id == 1):
        lst_lbl_1.append(png_no_ext)   
    if (lbl_id == 2):
        lst_lbl_2.append(png_no_ext)   
    if (lbl_id == -1):
        lst_lbl_minus_1.append(png_no_ext)   


# In[165]:


len(lst_lbl_0)


# In[166]:


len(lst_lbl_1)


# In[167]:


len(lst_lbl_2)


# In[168]:


len(lst_lbl_minus_1)


# In[132]:


import random


# In[155]:


v_lbl_0 = random.sample(lst_lbl_0, 273)


# In[156]:


v_lbl_1 = random.sample(lst_lbl_1, 103)


# In[157]:


v_lbl_2 = random.sample(lst_lbl_2, 47)


# In[158]:


v_lbl_minus_1 = random.sample(lst_lbl_minus_1, 167)


# In[161]:


image_to_move = v_lbl_0[1]


# In[170]:


# shutil.rmtree('./data-siim/valid/labels/')
# ! mkdir './data-siim/valid/labels/'
# shutil.rmtree('./data-siim/valid/images/')
# ! mkdir './data-siim/valid/images/'


# In[171]:


def move_files(image_to_move):
    try:
        fl_from_move = f'./data-siim/train/images/{image_to_move}.png'
        fl_to_move = f'./data-siim/valid/images/{image_to_move}.png'
        lbl_from_move = f'./data-siim/train/labels/{image_to_move}.txt'
        lbl_to_move = f'./data-siim/valid/labels/{image_to_move}.txt'
        shutil.move(fl_from_move, fl_to_move)
        shutil.move(lbl_from_move, lbl_to_move)
    except:
        print(image_to_move, " has problems in moving")


# In[172]:


for img in v_lbl_0:
    move_files(img)


# In[173]:


for img in v_lbl_1:
    move_files(img)


# In[174]:


for img in v_lbl_2:
    move_files(img)


# In[175]:


for img in v_lbl_minus_1:
    move_files(img)


# In[2]:


len(os.listdir('./data-siim/train/images/'))


# In[3]:


import os
import zipfile

WORKING_DIR = './data-siim/train/images/'
DEST_ZIP_FILE = 'yolo_train_images_0_1200.zip'

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files[0:1200]:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

zipf = zipfile.ZipFile( DEST_ZIP_FILE, 'w', zipfile.ZIP_DEFLATED)
zipdir(WORKING_DIR, zipf)
zipf.close()


# In[4]:


import logging
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

def save_to_bucket(bucket_name, source_file, dest_file):
    with open( source_file, "rb") as f:
        s3.upload_fileobj(f, bucket_name, dest_file)

src = "./yolo_train_images_0_1200.zip"
dest = "yolo_train_images_0_1200.zip"
bucket_name = "ritesh-s3"
save_to_bucket(bucket_name = bucket_name, source_file= src, dest_file = dest)


# In[11]:


import os
import zipfile

WORKING_DIR = './data-siim/train/labels/'
DEST_ZIP_FILE = 'yolo_train_labels_all.zip'

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

zipf = zipfile.ZipFile( DEST_ZIP_FILE, 'w', zipfile.ZIP_DEFLATED)
zipdir(WORKING_DIR, zipf)
zipf.close()


# In[12]:


import logging
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

def save_to_bucket(bucket_name, source_file, dest_file):
    with open( source_file, "rb") as f:
        s3.upload_fileobj(f, bucket_name, dest_file)

src = "./yolo_train_labels_all.zip"
dest = "yolo_train_labels_all.zip"
bucket_name = "ritesh-s3"
save_to_bucket(bucket_name = bucket_name, source_file= src, dest_file = dest)


# In[13]:


import os
import zipfile

WORKING_DIR = './data-siim/valid/images/'
DEST_ZIP_FILE = 'yolo_valid_images_all.zip'

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

zipf = zipfile.ZipFile( DEST_ZIP_FILE, 'w', zipfile.ZIP_DEFLATED)
zipdir(WORKING_DIR, zipf)
zipf.close()


# In[14]:


import logging
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

def save_to_bucket(bucket_name, source_file, dest_file):
    with open( source_file, "rb") as f:
        s3.upload_fileobj(f, bucket_name, dest_file)

src = "./yolo_valid_images_all.zip"
dest = "yolo_valid_images_all.zip"
bucket_name = "ritesh-s3"
save_to_bucket(bucket_name = bucket_name, source_file= src, dest_file = dest)


# In[15]:


import os
import zipfile

WORKING_DIR = './data-siim/valid/labels/'
DEST_ZIP_FILE = 'yolo_valid_labels_all.zip'

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))

zipf = zipfile.ZipFile( DEST_ZIP_FILE, 'w', zipfile.ZIP_DEFLATED)
zipdir(WORKING_DIR, zipf)
zipf.close()


# In[16]:


import logging
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

def save_to_bucket(bucket_name, source_file, dest_file):
    with open( source_file, "rb") as f:
        s3.upload_fileobj(f, bucket_name, dest_file)

src = "./yolo_valid_labels_all.zip"
dest = "yolo_valid_labels_all.zip"
bucket_name = "ritesh-s3"
save_to_bucket(bucket_name = bucket_name, source_file= src, dest_file = dest)


# In[ ]:





# In[ ]:


########################################################################################################################


# In[105]:


bbtxts = os.listdir('./data-siim/train/bbtexts/')
len(bbtxts)


# In[111]:


import logging
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

def save_to_bucket(bucket_name, source_file, dest_file):
    with open( source_file, "rb") as f:
        s3.upload_fileobj(f, bucket_name, dest_file)

src = "./train_bbtexts.zip"
dest = "train_bbtexts.zip"
bucket_name = "ritesh-s3"
save_to_bucket(bucket_name = bucket_name, source_file= src, dest_file = dest)


# In[5]:


import os
os.listdir("./yolov5/runs/train/exp5/weights/best.pt")


# In[6]:


import logging
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

def save_to_bucket(bucket_name, source_file, dest_file):
    with open( source_file, "rb") as f:
        s3.upload_fileobj(f, bucket_name, dest_file)

src = ("./yolov5/runs/train/exp5/weights/best.pt")
dest = "exp5-best.pt"
bucket_name = "ritesh-s3"
save_to_bucket(bucket_name = bucket_name, source_file= src, dest_file = dest)


# In[ ]:


# need to transfer some of the training files to validation dataset.


# In[ ]:


./runs/train/exp5/weights/best.pt


# In[ ]:





# In[ ]:





# In[107]:


df_merge[df_merge['cat_name'] != "none"].shape


# In[61]:


bbs[0]


# In[62]:


bb_xywh(bbs[0])


# In[55]:


plot_image_bb(im, bbs, lbl_id, figsize = (10,10))


# In[54]:





# In[ ]:





# In[ ]:


# DELETED BELOW THIS


# In[12]:


FULL_RUN = False
path = Path('./data-siim/train/')
path_im = path/'images'
path_lbl = path/'masks'


# In[13]:


fnames = get_image_files(path_im)
lbl_names = get_image_files(path_lbl)


# In[14]:


img_fn = fnames[10]
img = PILImage.create(img_fn)
img.show(figsize=(5,5));


# In[15]:


get_msk = lambda o: path/'masks'/f'{o.stem}{o.suffix}'


# In[16]:


msk = PILMask.create(get_msk(img_fn))
msk.show(figsize=(5,5), alpha=1);


# In[17]:


tensor(msk).unique()


# In[18]:


codes = np.array(['none','typical', 'indeterminate', 'atypical' ], dtype = str); codes


# In[19]:


lst_categories


# In[20]:


sz = msk.shape; sz
half = tuple(int(x/2) for x in sz); half


# In[21]:


# camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
#                    get_items=get_image_files,
#                    splitter=FileSplitter(path/'valid.txt'),
#                    get_y=get_msk,
#                    batch_tfms=[*aug_transforms(size=half), Normalize.from_stats(*imagenet_stats)])


# In[22]:


siimblock = DataBlock(blocks=(ImageBlock, MaskBlock(codes = codes)),
    get_items=get_image_files,
    splitter=RandomSplitter(),
    get_y= get_msk,
    item_tfms=Resize(768, method = "squish"),
    batch_tfms= [Normalize.from_stats(*imagenet_stats)]
    )

dls = siimblock.dataloaders(path/'images', bs=2)


# In[23]:


def get_subset_data(path):
    return fnames[0:500]


# In[23]:





# In[27]:


dls.show_batch(max_n=4, vmin=1, vmax=30, figsize=(14,10))


# In[24]:


dls.vocab = codes


# In[25]:


small_dls.vocab = codes


# In[26]:


name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['none']; void_code


# In[31]:


lst_categories


# In[27]:


name2id


# In[28]:


# void_code = name2id['none']; void_code


# In[29]:


def acc_simvid(inp, targ):
  targ = targ.squeeze(1)
  mask = targ != void_code
  return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()


# In[30]:


opt = ranger


# In[31]:


learn = unet_learner(dls, resnet34, metrics=acc_simvid, self_attention=True, act_cls=Mish, opt_func=opt)


# In[30]:


learn.lr_find()


# In[41]:


lr = 1e-3


# In[45]:


siimblock_small = DataBlock(blocks=(ImageBlock, MaskBlock(codes = codes)),
    get_items=get_subset_data,
    splitter=RandomSplitter(),
    get_y= get_msk,
    item_tfms=Resize(768, method = "squish"),
    batch_tfms= [Normalize.from_stats(*imagenet_stats)]
    )

small_dls = siimblock_small.dataloaders(path/'images', bs=2)
learn.dls = small_dls


# In[46]:


learn.fit_flat_cos(1, slice(lr))


# In[47]:


#learn.save("siim-seg-002-resnet34")


# In[37]:


#learn = learn.load("siim-seg-001-resnet34")


# In[48]:


learn.show_results()


# In[50]:


learn.dls = dls


# In[51]:


learn.fit_flat_cos(2, slice(lr))
# learn.save("siim-seg-003-resnet34")


# In[52]:


learn.show_results()


# In[ ]:


lrs = slice(lr/400, lr/4)
learn.unfreeze()
learn.fit_flat_cos(2 , lrs)
# learn.save("siim-seg-004-resnet34")


# In[32]:


learn = learn.load("siim-seg-004-resnet34")


# In[33]:


learn.show_results()


# In[ ]:




