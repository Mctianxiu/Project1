
# 根据图片名称生成 train.txt test.txt

import os
import random

# train_file = open(R'C:\Users\YuanBao\Desktop\Enhanced_Results\UIQS-3630/UIQS-100.txt', 'w')
test_file = open(R'C:\Users\YuanBao\Desktop\1\WallEye.txt', 'w')
#
# for _, _, train_files in os.walk(R'F:\Compare_Data\UIQS-3630'):
#     continue
for _, _, test_files in os.walk(R'C:\Users\YuanBao\Desktop\1\WallEye'):          # 遍历指定路径下的所有文件和子文件夹
    random.shuffle(test_files)                                                   # 打乱文件名顺序，确保测试集的随机性
    continue

# for file in train_files:
#     name = ['_name_UIQS-100-Dewater-baseline_8_100',
#             '_name_UIQS-100-Dewater-baseline_8_200',
#             '_name_UIQS-100-Dewater-baseline_9',
#             '_name_UIQS-100-Dewater-baseline_CycleGAN',
#             '_name_UIQS-100-Dewater-baseline_Improved Model',
#             '_name_UIQS-100-Dewater-baseline_Physical_Li',
#             '_name_UIQS-100-Dewater-baseline_UWGAN_UIE',
#             '_name_UIQS-100-Dewater-baseline_WaterGAN']
#     for img_name in name:
#         train_file.write(file.split('.bmp')[0] + img_name + '.bmp' + '\n')

for file in test_files:
    test_file.write(file+'\n')                  # 将每个文件名写入 WallEye.txt，每行一个文件名
