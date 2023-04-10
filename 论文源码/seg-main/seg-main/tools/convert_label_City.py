# import cv2
# import os
# import numpy as np
#
# SrcPath = '/home/rarabura/datasets/KITTI_road/leftImg8bit/train_0'
# SavePath = '/home/rarabura/datasets/KITTI_road/leftImg8bit/train'
# # MaskPath = 'G:/cv2_Freespace/20211207/666666.jpg'
#
# img_list = os.listdir(SrcPath)
# if not os.path.exists(SavePath):
#     os.mkdir(SavePath)
#
# # mask = cv2.imread(MaskPath)[:, :, 0]
# # print("-----mask label----------", np.unique(mask))  # 1 255
#
# for img in img_list:
#     label_path = SrcPath + '/' + img
#     # save_path = SavePath + '/' + img[:-4] + '_gtFine_labelTrainIds.png'
#     save_path = SavePath + '/' + img[:-4] + '_leftImg8bit.png'
#     label0 = cv2.imread(label_path)
#     # label = cv2.imread(label_path)[:, :, 0]
#     # label = cv2.resize(label, (480, 640), interpolation=cv2.INTER_NEAREST)
#     # new_label = np.zeros_like(label)
#     # print(np.unique(label))
#
#     # new_label[label == 0] = 0
#     # new_label[label == 2] = 2
#     # new_label[label == 3] = 3
#     # new_label[label == 4] = 4
#     # new_label[label == 5] = 5
#     # new_label[label == 6] = 6
#
#     # new_label[label == 255] = 1
#     # new_label[mask==255] = 0
#
#     # point_left_uper = (216,257)
#     # point_right_lower = (262,383)
#     # (216,257) (262,389) -> [257:389, 216:262]
#     # new_label[point_left_uper[1]:point_right_lower[1], point_left_uper[0]:point_right_lower[0]] = 255
#
#     cv2.imwrite(save_path, label0)
#     print(np.unique(label0))
#     print(save_path)

import glob
import shutil
import os

print("start job")
img_list = glob.glob('/home/rarabura/datasets/KITTI_road/gtFine/train_0/*.png')
print("Get img list finished!")

for img in img_list:
    if os.path.exists(img):
        save_img = img[:-4] + '_gtFine_labelTrainIds.png'
        # save_img = img[:-4] + '_leftImg8bit.png'
        shutil.copyfile(img, save_img)
        print(save_img, " finished !")
print("Done!")

