from sn6 import Sn6Create

ds_path = 'D:/Projects/ml-comp/spacenet6-challenge/train/AOI_11_Rotterdam/'
mode = 'SAR-Intensity'
sar_ch = [1,4,3]
img_size = (256,256)
sn6 = Sn6Create(ds_path, None, mode, img_size, sar_ch)