import torch
from torch import nn
from PIL import Image
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os

class LevirCd(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode="train"):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode

        file_data = []
        train_dir = os.path.join(root_dir, self.mode)
        A_dir = os.path.join(train_dir, 'A')
        B_dir = os.path.join(train_dir, 'B')
        label_dir = os.path.join(train_dir, 'label')

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

    def __len__(self):
        return len(self.a_file)

    def __getitem__(self, idx):
        img_a = Image.open(os.path.join(self.A_dir, self.a_file[idx]))
        img_b = Image.open(os.path.join(self.B_dir, self.b_file[idx]))
        img_l = Image.open(os.path.join(self.label_dir, self.l_file[idx]))

        # Skip Connection 改造不需要动 Dataset 部分
        # 但训练时需要 RandomCrop 同步裁剪（之前修过的bug）
        i, j, h, w = transforms.RandomCrop.get_params(img_a, output_size=(256, 256))
        img_a = transforms.functional.crop(img_a, i, j, h, w)
        img_b = transforms.functional.crop(img_b, i, j, h, w)
        img_l = transforms.functional.crop(img_l, i, j, h, w)

        to_tensor = transforms.ToTensor()
        return to_tensor(img_a), to_tensor(img_b), to_tensor(img_l)


class Encoder(nn.Module):
    """
    编码器：把 256x256 的图像压缩成 4 个不同尺度的特征图
    
    旧版：用一个大 nn.Sequential，只输出最后一层 16x16
    新版：拆成 4 个 stage，每层都输出，供 Decoder skip 使用
    
    Skip Connection 的核心思想：
    - 浅层（e1/e2）包含边缘、纹理等细节信息 → 帮助画精确边界
    - 深层（e3/e4）包含语义信息（"这是房子"、"这是路"）→ 帮助判断哪里变了
    - Decoder 同时拿到这两种信息 → 预测既准确又精细
    """

    def __init__(self):
        super().__init__()

        # Stage 1: 256x256 -> 128x128, 64通道
        # 输入是 RGB 图像(3通道)，输出 64 通道的特征图
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 尺寸减半: 256->128
        )

        # Stage 2: 128x128 -> 64x64, 128通道
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 128->64
        )

        # Stage 3: 64x64 -> 32x32, 256通道
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 64->32
        )

        # Stage 4: 32x32 -> 16x16, 512通道
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 32->16
        )

    def forward(self, x):
        e1 = self.stage1(x)   # [B,  64, 128, 128]  ← 最浅层，细节最丰富
        e2 = self.stage2(e1)  # [B, 128,  64,  64]
        e3 = self.stage3(e2)  # [B, 256,  32,  32]
        e4 = self.stage4(e3)  # [B, 512,  16,  16]  ← 最深层，语义最强
        return e1, e2, e3, e4


class SiameseFCN(nn.Module):
    """
    孪生全卷积网络 + Skip Connection (U-Net风格)
    
    数据流：
    T1图像 ──> Encoder ──> e1,e2,e3,e4
                                      ↓
    T2图像 ──> Encoder ──> e1,e2,e3,e4
                                      ↓
                              |f1-f2| 差异 (对4层分别算差异！)
                                      ↓
              Decoder 逐层上采样 + cat(skip特征) → 最终预测图
    
    关键变化（对比旧版）：
    旧版：Encoder 只输出 1 层(16x16) → Decoder 盲猜细节
    新版：Encoder 输出 4 层 → Decoder 每层都能"看到"对应尺度的细节
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        # ======== Decoder 各层 ========
        # 每层先上采样，然后在外面和 skip 特征 cat
        # 所以 Conv2d 的入通道数 = 该层的实际输入通道数（还没cat之前！）

        # Up1: 输入=diff4(512ch,16x16) → 上采样→Conv→输出256ch
        #       然后在外部 cat diff3(256ch) → 变成512ch 给下一层
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(),   # 输入=diff4的512ch
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU()
        )

        # Up2: 输入=d1_cat(256+256=512ch,32x32) → 上采样→Conv→输出128ch
        #       然后在外部 cat diff2(128ch) → 变成256ch 给下一层
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(),   # 输入=cat后的512ch
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU()
        )

        # Up3: 输入=d2_cat(128+128=256ch,64x64) → 上采样→Conv→输出64ch
        #       然后在外部 cat diff1(64ch) → 变成128ch 给下一层
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(),   # 输入=cat后的256ch
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU()
        )

        # Up4: 输入=d3_cat(64+64=128ch,128x128) → 上采样→Conv→输出32ch
        #       不再cat了，直接给final层
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),    # 输入=cat后的128ch
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU()
        )

        # 最终输出：32ch → 1ch（二分类：变化/无变化）
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        # === Step 1: 双分支编码器提取特征 ===
        e1_1, e2_1, e3_1, e4_1 = self.encoder(x1)
        e1_2, e2_2, e3_2, e4_2 = self.encoder(x2)

        # === Step 2: 计算每层的差异特征（关键！不只是最后一层） ===
        diff1 = torch.abs(e1_1 - e1_2)   # [B,  64, 128, 128]  边缘/纹理差异
        diff2 = torch.abs(e2_1 - e2_2)   # [B, 128,  64,  64]
        diff3 = torch.abs(e3_1 - e3_2)   # [B, 256,  32,  32]
        diff4 = torch.abs(e4_1 - e4_2)   # [B, 512,  16,  16]  语义差异

        # === Step 3: 解码器逐层上采样 + Skip Connection ===
        # Skip Connection = 把 encoder 的浅层特征 "跳过" 中间层，直接送给 decoder
        # 用 torch.cat 在通道维度(dim=1)拼接

        d1 = self.up1(diff4)                       # [B, 256, 32,  32]
        d1 = torch.cat([d1, diff3], dim=1)         # [B, 256+256=512, 32, 32] ← skip!

        d2 = self.up2(d1)                          # [B, 128, 64,  64]
        d2 = torch.cat([d2, diff2], dim=1)         # [B, 128+128=256, 64, 64] ← skip!

        d3 = self.up3(d2)                          # [B,  64, 128, 128]
        d3 = torch.cat([d3, diff1], dim=1)         # [B,  64+64=128, 128, 128] ← skip!

        out = self.up4(d3)                         # [B,  32, 256, 256]
        out = self.final(out)                      # [B,   1, 256, 256]
        out = self.sigmoid(out)                    # [B,   1, 256, 256]  值域0~1
        return out


# 验证代码
if __name__ == '__main__':
    net = SiameseFCN()
    a = torch.randn(2, 3, 256, 256)
    b = torch.randn(2, 3, 256, 256)
    print(net(a, b).shape)  # 应该输出: torch.Size([2, 1, 256, 256])

    # 统计参数量
    total_params = sum(p.numel() for p in net.parameters())
    print(f'模型参数量: {total_params:,} ({total_params/1000:.1f}K)')
