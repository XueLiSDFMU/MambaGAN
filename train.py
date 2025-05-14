
import torch
import time
import os
import numpy as np
weight_dir = "weights/"
if not os.path.exists(weight_dir): #如果不存在权重文件，自动创建
    os.makedirs(weight_dir)

train_MRI_image_file_dir = 'brain_Dataset/train_data_new/MRI/'
train_CT_image_file_dir = 'brain_Dataset/train_data_new/CT/'
train_MRI_image_txt_file_dir = 'brain_Dataset/txt_file_new/train_MRI.txt'
train_CT_image_txt_file_dir = 'brain_Dataset/txt_file_new/train_CT.txt'

train_MRI_image_file = open(train_MRI_image_txt_file_dir)  # 训练数据的名字放到txt文件里
train_MRI_image_strings = train_MRI_image_file.readlines()
train_CT_image_file = open(train_CT_image_txt_file_dir)  # 训练数据的名字放到txt文件里
train_CT_image_strings = train_CT_image_file.readlines()
generator =MambaGAN()
discriminator = Discriminator()
d_loss_function = torch.nn.BCELoss().to(device=device)
g_loss_function = SSIM3D().to(device=device)
g_residual_loss_function = torch.nn.MSELoss().to(device=device)

for epoch in range(20)
 MR_image_patch1 = MR_image_patch[i]
            CT_image_patch1 = CT_image_patch[i]
            MR_image_patch1 = np.array(MR_image_patch1)
            CT_image_patch1 = np.array(CT_image_patch1)
            MR_image_patch1 = MR_image_patch1[np.newaxis, np.newaxis, ...]
            CT_image_patch1 = CT_image_patch1[np.newaxis, np.newaxis, ...]
            MR_image_patch1 = torch.tensor(MR_image_patch1)
            CT_image_patch1 = torch.tensor(CT_image_patch1)
            MR_image_patch1 = MR_image_patch1.to(device)
            CT_image_patch1 = CT_image_patch1.to(device)
            optimizer_D.zero_grad()
            # 生成伪CT图像
            sCT_image4,sCT_image8,sCT_image16,sCT_image32,sCT_image = generator(MR_image_patch1)

            ##使用判别器识别伪CT图像并输出判别结果
            CT_real_prob = discriminator(CT_image_patch1)
            d_loss_real = d_loss_function(CT_real_prob,valid)
            ##4
            CT_fake_prob4 = discriminator(sCT_image4)
            d_loss_fake4 = d_loss_function(CT_fake_prob4,fake)
            ##8
            CT_fake_prob8 = discriminator(sCT_image8)
            d_loss_fake8 = d_loss_function(CT_fake_prob8, fake)
            ##16
            CT_fake_prob16 = discriminator(sCT_image16)
            d_loss_fake16 = d_loss_function(CT_fake_prob16, fake)
            ##32
            CT_fake_prob32 = discriminator(sCT_image32)
            d_loss_fake32 = d_loss_function(CT_fake_prob32, fake)
            ##64
            CT_fake_prob = discriminator(sCT_image)
            d_loss_fake = d_loss_function(CT_fake_prob, fake)


            d_loss = (d_loss_real + d_loss_real + d_loss_real + d_loss_real + d_loss_real + d_loss_fake4 +d_loss_fake8 + d_loss_fake16 + d_loss_fake32 + d_loss_fake)/5
            ##优化判别器

            ##生成器估计
            optimizer_G.zero_grad()
            g_loss4 = g_loss_function(CT_image_patch1,sCT_image4)
            g_loss8 = g_loss_function(CT_image_patch1,sCT_image8)
            g_loss16 = g_loss_function(CT_image_patch1,sCT_image16)
            g_loss32 = g_loss_function(CT_image_patch1,sCT_image32)
            g_loss = g_loss_function(CT_image_patch1,sCT_image)
            predict_residual = sCT_image - CT_image_patch1
            residual_loss  = g_residual_loss_function(residual,predict_residual)
   
            #writer.add_scalar('running_loss', g_loss, d_loss, global_step=((epoch * 600) + (i + 1)))
            step = step + 1
            print( '[%d, %5d] g_loss4: %.5f g_loss8: %.5f g_loss16: %.5f g_loss32: %.5f g_loss: %.5f  residual_loss: %.5f d_loss: %.5f'
                  % (epoch + 1, step, g_loss4,g_loss8,g_loss16,g_loss32,g_loss,residual_loss,d_loss))  # 然后再除以200，就得到这两百次的平均损失值
            # 每50个epoch保存一次参数
            if step % 10000==0:
                torch.save(generator.state_dict(), weight_dir + str(epoch) + '_'+str(step) +"_net_params" + ".pkl")

        torch.save(generator.state_dict(), weight_dir + str(epoch) + "_net_params" + ".pkl")

    print('Finished Training')
    # 保存神经网络
    torch.save(generator, 'net.pkl')  # 保存整个神经网络的结构和模型参数
    # torch.save(net.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数
