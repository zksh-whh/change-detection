from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import os


class LevirCd(torch.utils.data.Dataset):

#     def __init__(self, root_dir, split='train'):
#         super().__init__()
#         self.root_dir = root_dir
#         self.split = split
#         dataset_list = []
#
#         data_dir = os.path.join(root_dir, self.split)
#         data_A_dir = os.path.join(data_dir, 'A')
#         data_B_dir = os.path.join(data_dir, 'B')
#         data_label_dir = os.path.join(data_dir, 'label')
#         self.A_dir = data_A_dir
#         self.B_dir = data_B_dir
#         self.label_dir = data_label_dir
#
#         dataset_list.append(os.listdir(data_A_dir))
#         dataset_list[0].sort()
#         dataset_list.append(os.listdir(data_B_dir))
#         dataset_list[1].sort()
#         dataset_list.append(os.listdir(data_label_dir))
#         dataset_list[2].sort()
#
#         #self.A_files = dataset_list[0] #我理解的这里我传入的是数据;# 这里存的是"文件名列表"，不是图像数据
#         self.A_files =dataset_list[0]
#         self.B_files = dataset_list[1]
#         self.label_files = dataset_list[2]
#         self.transform = transforms.ToTensor()
#
#     def __len__(self):
#         return len(self.A_files)
#
#     def __getitem__(self,idx):
#
#         img_a = Image.open(os.path.join(self.A_dir, self.A_files[idx]))
#         img_b = Image.open(os.path.join(self.B_dir, self.B_files[idx]))
#         img_label = Image.open(os.path.join(self.label_dir,self.label_files[idx]))
#
#         return self.transform(img_a), self.transform(img_b), self.transform(img_label)
#
#     def show(self):
#         img_a, img_b, img_label = self[0]
#         fig,axes = plt.subplots(1,3,figsize=(15,5))
#
#         axes[0].imshow(img_a.permute(1,2,0))
#         axes[0].set_title('T1')
#         axes[0].axis('off')
#
#         axes[1].imshow(img_b.permute(1,2,0))
#         axes[1].set_title('T2')
#         axes[1].axis('off')
#
#         axes[2].imshow(img_label.squeeze(),cmap='gray')
#         axes[2].set_title('Change Label')
#         axes[2].axis('off')
#
#         plt.tight_layout()
#         plt.show()
#
#
# #root_dir =
# ds = LevirCd(r"E:\python\Deep_learn\d2l\data\archive\LEVIR_CD",split='val')
# print(len(ds))
# ds.show()







    def __init__(self,root_dir,mode ="train"):
        #父类继承前面不要忘记括号啊！！！
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode

        file_data = []
        train_dir = os.path.join(root_dir,self.mode)
        A_dir = os.path.join(train_dir,'A')
        B_dir = os.path.join(train_dir,'B')
        label_dir = os.path.join(train_dir,'label')

        file_data.append(os.listdir(A_dir))
        file_data[0].sort()
        file_data.append(os.listdir(B_dir))
        file_data[1].sort()
        file_data.append(os.listdir(label_dir))
        file_data[2].sort()

        self.A_dir = A_dir
        self.B_dir = B_dir
        self.label_dir = label_dir

        self.a_file = file_data[0]
        self.b_file = file_data[1]
        self.l_file = file_data[2]
        self.crop_size = 256
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.a_file)

    def __getitem__(self,idx):
        img_a = Image.open(os.path.join(self.A_dir,self.a_file[idx]))
        img_b = Image.open(os.path.join(self.B_dir,self.b_file[idx]))
        img_l = Image.open(os.path.join(self.label_dir,self.l_file[idx]))

        # 关键修复：先获取同一次随机裁剪参数，再对三张图用同一位置裁剪！
        i, j, h, w = transforms.RandomCrop.get_params(img_a, output_size=(self.crop_size, self.crop_size))
        img_a = transforms.functional.crop(img_a, i, j, h, w)
        img_b = transforms.functional.crop(img_b, i, j, h, w)
        img_l = transforms.functional.crop(img_l, i, j, h, w)

        return self.to_tensor(img_a), self.to_tensor(img_b), self.to_tensor(img_l)
