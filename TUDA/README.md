# TUDA
This is the PyTorch implementation for our TIP'2023 paper:

'Domain Adaptation for Underwater Image Enhancement'


## Environment
1. Python 3.8.15
2. PyTorch 1.7.1
3. CUDA 10.1.243
4. Cudnn 7.6.5
5. Ubuntu 18.04 / Window 10.1


## Test
step 1： 
use generate_train_or_test_txt.py to generate xxx.txt from your test dataset xxx
 
step 2:
```
########## change test_num_id based on your test dataset name ###########
Test_global.py   
pix2pix_class_test_Deater.py
Test_SingleGAN.py

```
step 3: 
python Test_SingleGAN.py
 
## Note ##
The result in the 'Image_XXX' folder is the final result of our TUDA.

The result in the 'Inetr_XXX' folder is the result of our interdomain phase, not the final result. Please be careful to avoid confusion.

## Contact
If you have any questions, please contact: Zhengyong Wang: zywang@shu.edu.cn
