from pathlib import Path
import os
import glob
import cv2
import png
import numpy as np
import shutil
from PIL import Image


# set add_4 and add_15 folder
seg_folder = Path('./20210311_ground_mask_part1/')

seg_folder_TrainID = Path(os.path.join(seg_folder,"TrainID"))
seg_folder_img = Path(os.path.join(seg_folder,"img"))
seg_folder_LabelID = Path(os.path.join(seg_folder,"LabelID"))
seg_folder_color = Path(os.path.join(seg_folder,"color"))

if not seg_folder_img.exists():
    os.mkdir(seg_folder_img)
if not seg_folder_LabelID.exists():
    os.mkdir(seg_folder_LabelID)
if not seg_folder_TrainID.exists():
    os.mkdir(seg_folder_TrainID)
if not seg_folder_color.exists():
    os.mkdir(seg_folder_color)

LabelID_glob = glob.glob('./20210311_ground_mask_part1/20210222_TJP_freespace_ss/20210222_TJP_freespace_ss_label/*.png')
TrainID_glob = glob.glob('./20210311_ground_mask_part1/20210222_TJP_freespace_ss/20210222_TJP_freespace_ss_label/*.png')
Img_glob = glob.glob('./20210311_ground_mask_part1/20210222_TJP_freespace_ss/20210222_TJP_freespace_ss_extract/*.jpg')
Color_glob = glob.glob('./20210311_ground_mask_part1/20210222_TJP_freespace_ss/20210222_TJP_freespace_ss_label/*.png')
# assert(len(LabelID_glob)==len(Img_glob))

print("len for lable glob",len(LabelID_glob))

# ******************* TrainID process ****************************
print("begin to process TrainID")
for k in range(len(LabelID_glob)):
    transfer_ori = Image.open(TrainID_glob[k])
    transfer_ground = np.array(transfer_ori)
    transfer_ground[transfer_ground == 0] = 255  # ignore
    transfer_ground[transfer_ground == 1] = 0   # freespace
    transfer_ground[transfer_ground == 2] = 1  # white solid lane line
    transfer_ground[transfer_ground == 3] = 2  # white dotted lane line
    # transfer_ground[transfer_ground == 4] = 3  # yellow solid lane line
    # transfer_ground[transfer_ground == 5] = 4  # yellow dotted lane line
    transfer_ground[transfer_ground == 6] = 3  # arrow
    transfer_ground[transfer_ground == 7] = 4  # diamond_sign
    transfer_ground[transfer_ground == 8] = 5  # zebra crossing
    transfer_ground[transfer_ground == 9] = 6  # stop line
    transfer_ground_img = Image.fromarray(transfer_ground)
    transfer_ground_img = transfer_ground_img.resize((2048, 1024))
    transfer_ori_path = os.path.join(seg_folder_TrainID,TrainID_glob[k].split('/')[-1].split('\\')[1])
    transfer_ground_img.save(transfer_ori_path)
    print("the {0} th TrainID img has been processed and save in folder".format(k))
#
# # ******************* LableID process ****************************
print("begin to process LableID")
for k in range(len(LabelID_glob)):
    transfer_ori = Image.open(TrainID_glob[k])
    transfer_ground = np.array(transfer_ori)
    transfer_ground[transfer_ground == 0] = 0  # ignore
    transfer_ground[transfer_ground == 1] = 1  # freespace
    transfer_ground[transfer_ground == 2] = 2  # white solid lane line
    transfer_ground[transfer_ground == 3] = 3  # white dotted lane line
    # transfer_ground[transfer_ground == 4] = 4  # yellow solid lane line
    # transfer_ground[transfer_ground == 5] = 5  # yellow dotted lane line
    transfer_ground[transfer_ground == 6] = 4  # arrow
    transfer_ground[transfer_ground == 7] = 5  # diamond_sign
    transfer_ground[transfer_ground == 8] = 6  # zebra crossing
    transfer_ground[transfer_ground == 9] = 7  # stop line

    transfer_ground_img = Image.fromarray(transfer_ground)
    transfer_ground_img = transfer_ground_img.resize((2048, 1024))
    transfer_ori_path = os.path.join(seg_folder_TrainID, TrainID_glob[k].split('/')[-1].split('\\')[1])
    transfer_ground_img.save(transfer_ori_path)
    print("the {0} th LabelID img has been processed and save in folder".format(k))


# # ******************** resize img ***********************************
for k in range(len(Img_glob)):
    print("copy the {0}th img to add img folder".format(k))
    src_img = Image.open(Img_glob[k])
    src_img = src_img.resize((2048, 1024))
    src_img_save_path = os.path.join(seg_folder_img,Img_glob[k].split('/')[-1].split('\\')[1].split('.')[0])
    src_img.save(src_img_save_path+'.png')
#
# ## ********************* resize color png *****************************
for k in range(len(Color_glob)):
    print("copy the {0}th img to color folder".format(k))
    src_img = Image.open(Color_glob[k])
    src_img = src_img.resize((2048,1024))
    color_img_save_path = os.path.join(seg_folder_color,Color_glob[k].split('/')[-1].split('\\')[1].split('.')[0])
    src_img.save(color_img_save_path+'.png')
