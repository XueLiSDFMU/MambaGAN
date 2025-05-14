import pickle
import torch
import nibabel as nib
from skimage.metrics import structural_similarity,normalized_mutual_information,mean_squared_error,peak_signal_noise_ratio
from functions.patch_concat.patch_weight import *
from functions.utils.patch_operations import slice_3Dmatrix,concat_3Dmatrices
from write_excel import write_excel_xls_append



def generator_test_data(image_file,st,label_file,sl,mask_file,sm):
    vol_dir = image_file + st
    image1 = nib.load(vol_dir)
    image = image1.get_fdata()
    affine0 = image1.affine.copy()
    image = np.asarray(image, dtype=np.float32)
    label_dir = label_file+sl
    label = nib.load(label_dir).get_fdata()
    label = np.asarray(label, dtype=np.float32)
    affine0 = np.asarray(affine0, dtype=np.float32)
    mask_dir = mask_file+sm
    mask = nib.load(mask_dir).get_fdata()
    mask = np.asarray(mask, dtype=np.float32)
    return image,label,mask, affine0
def gen_patch(moving_image, fixed_image):
    moving_image_patch = slice_3Dmatrix(moving_image, window=(64, 64, 64), overlap=(32, 32, 48))
    fixed_image_patch = slice_3Dmatrix(fixed_image, window=(64, 64, 64), overlap=(32, 32, 48))
    return moving_image_patch, fixed_image_patch

def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

def Max_min_normalization(image):
    [x, y, z] = np.shape(image)
    image = np.reshape(image, (x * y * z, 1))
    max = np.max(image)
    min = np.min(image)
    image = (image - min) / (max - min)
    image = np.reshape(image, (x, y, z))
    return image, max, min
patch_weight = get_loss_3D_distance_weight((64,64,64), mode=0, stride=0)  ##mode=[0,1,2,3,4]
def voxel_to_CT_value(image, max, min):
    [x, y, z] = np.shape(image)
    image = np.reshape(image, (x * y * z, 1))
    image = image * (max - min) + min
    image = np.reshape(image, (x, y, z))
    return image


    # 加载数据
test_MRI_image_file =  'brain_Dataset/MRItest/'
test_CT_image_file = 'brain_Dataset/CTtest/'
test_label_file = 'brain_Dataset/mask/'
##
test_MRI_txt_file = 'brain_Dataset/txt_file_new/MRtest.txt'
test_CT_txt_file = 'brain_Dataset/txt_file_new/CTtest.txt'
test_label_txt_file = 'brain_Dataset/txt_file_new/mask.txt'
MRI_image_file = open(test_MRI_txt_file)  # 训练数据的名字放到txt文件里
MRI_image_strings = MRI_image_file.readlines()
CT_image_file = open(test_CT_txt_file)
CT_image_strings = CT_image_file.readlines()
label_image_file = open(test_label_txt_file)
label_image_strings= label_image_file.readlines()
# 加载模型

model = MambaGAN()


model.load_state_dict(torch.load("weightlc/")) # 导入网络的参数
model.to(device=device)
for id in range(20):
    MRI_name = MRI_image_strings[id].strip()  # 文件名
    CT_name = CT_image_strings[id].strip()
    label_name = label_image_strings[id].strip()
    MRI_image, CT_image, label, affine = generator_test_data(test_MRI_image_file, MRI_name, test_CT_image_file, CT_name,
                                                             test_label_file, label_name)
    CT_image_BN, CT_max, CT_min = Max_min_normalization(CT_image)
    CT_image_BN = CT_image_BN * label
    mask = np.zeros((x,y,z))
    CT_image_BN = CT_image_BN*mask
    [W, H, C] = np.shape(MRI_image)
    MRI_image_path, CT_image_patch = gen_patch(MRI_image, CT_image_BN)
    predict_CTs = np.empty((len(MRI_image_path), 64, 64, 64))
    predict_ones = np.empty((len(MRI_image_path), 64, 64, 64))
    for i in range(len(MRI_image_path)):
        input_MRI_image = MRI_image_path[i]
        input_MRI_image = input_MRI_image[np.newaxis,np.newaxis, ...]
        input_MRI_image = torch.tensor(input_MRI_image)
        input_MRI_image = input_MRI_image.to(device)
        outputs = model(input_MRI_image)
        outputs = outputs[4]
        output = outputs.cpu()  # 将张量从CUDA设备转移到CPU
        predicts = output.detach().numpy()
        predicts = np.squeeze(predicts)
        predict_CTs[i,...] = predicts*patch_weight
        predict_ones[i,...] = patch_weight
    sCT = concat_3Dmatrices(predict_CTs, [W, H, C] ,[64,64,64], [32,32,48])
    count = concat_3Dmatrices(predict_ones, [W, H, C] ,[64,64,64], [32,32,48])
    pCT = sCT/count
    predict_CT_image = np.array(pCT, dtype=np.float32)
    predict_CT_image_BN, MAX1, MIN1 = Max_min_normalization(predict_CT_image)
    sCT_image_BN = predict_CT_image_BN*mask
    mse = mean_squared_error(CT_image_BN, sCT_image_BN)
    psnr = peak_signal_noise_ratio(CT_image_BN, sCT_image_BN)
    ssim = structural_similarity(CT_image_BN, sCT_image_BN,data_range=CT_image_BN.max() - CT_image_BN.min())
    mutual_information = normalized_mutual_information(CT_image_BN, sCT_image_BN)
    save_sCT_image = voxel_to_CT_value(sCT_image_BN, CT_max, CT_min)

    metric = [[np.float64(mse), np.float64(psnr), np.float64(ssim), np.float64(mutual_information)], ]
    print(metric)
    write_excel_xls_append('predicts/results.xls', metric)
