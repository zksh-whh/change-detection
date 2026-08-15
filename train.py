import torch
import os
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from Dataset import LevirCd
from model import SiameseFCN

batch_size =2
lr = 1e-4
epochs =100  # 增加到20轮，Dice Loss需要更多时间收敛
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_levir = LevirCd(r"E:\python\Deep_learn\d2l\data\archive\LEVIR_CD",mode='train')
test_levir = LevirCd(r"E:\python\Deep_learn\d2l\data\archive\LEVIR_CD",mode='test')
train_loader = DataLoader(train_levir,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_levir,batch_size=batch_size,shuffle=True)


net = SiameseFCN().to(device)

# ========== Dice Loss（解决类别不平衡：逼模型关注稀有的白色像素）==========
class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # pred和target都是 [B, 1, H, W]，值在0-1之间
        smooth = 1e-6  # 防止除零
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice  # dice越接近1越好，所以loss=1-dice

# 组合损失：BCE保证基础分类，Dice强制关注变化区域
bce_fn = nn.BCELoss()
dice_fn = DiceLoss()

def combined_loss(pred, target):
    return bce_fn(pred, target) + dice_fn(pred, target)

optimizer = torch.optim.Adam(net.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 每20轮学习率减半（100轮共减4-5次）
# print("device =", device)
# print("net 在哪？", next(net.parameters()).device)
# print("encoder 在哪？", next(net.encoder.parameters()).device)

for epoch in range(epochs):
    net.train()
    train_loss, train_correct, train_total = 0.0,0,0
    for i ,(a,b,l)  in enumerate(train_loader):
        a,b,l = a.to(device),b.to(device),l.to(device)
        optimizer.zero_grad()
        pred = net(a,b)
        loss = combined_loss(pred, l)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Epoch {epoch}, batch {i},Loss {loss.item() :.4f}')

        train_loss += loss.item()*l.numel()
        pred_binary = (pred > 0.5).float()
        train_correct += (pred_binary==l).sum().item()
        train_total += l.numel()
    print("训练完成！")
    net.eval()
    with torch.no_grad():
        # 预测时用 CenterCrop（固定切中心），避免 RandomCrop 随机到空白区域
        idx = 215  # train_293.png 的真实索引

        img_a = Image.open(os.path.join(train_levir.A_dir, train_levir.a_file[idx]))
        img_b = Image.open(os.path.join(train_levir.B_dir, train_levir.b_file[idx]))
        img_l = Image.open(os.path.join(train_levir.label_dir, train_levir.l_file[idx]))

        center_crop = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        a, b, l = center_crop(img_a), center_crop(img_b), center_crop(img_l)
        # 【修复1】关闭PIL图像释放内存（每轮泄漏3张图×100轮=300张）
        img_a.close(); img_b.close(); img_l.close()

        print(f'样本: {train_levir.a_file[idx]}, 变化占比: {l.mean():.4f}')

        a,b,l = a.unsqueeze(0).to(device),b.unsqueeze(0).to(device),l.unsqueeze(0).to(device)
        pred = net(a,b)

        fig, axes = plt.subplots(1,4,figsize=(20,5))
        axes[0].imshow(a[0].cpu().permute(1,2,0)); axes[0].set_title('T1');axes[0].axis('off')
        axes[1].imshow(b[0].cpu().permute(1,2,0)); axes[1].set_title('T2');axes[1].axis('off')
        axes[2].imshow(l[0,0].cpu(),cmap='gray'); axes[2].set_title(f'GT (变化{l.mean()*100:.1f}%)');axes[2].axis('off')
        axes[3].imshow(pred[0,0].cpu()>0.5, cmap='gray');axes[3].set_title('Prediction'); axes[3].axis('off')
        plt.tight_layout()
        plt.savefig('prediction_result.png', dpi=150)
        # 【修复2】关闭matplotlib figure释放内存（每轮泄漏1个figure×100轮=100个）
        plt.close(fig)
        # 调试：打印预测值的实际范围
        print(f'  pred范围: [{pred.min():.4f}, {pred.max():.4f}], 均值: {pred.mean():.4f}')
        print(f'预测图已保存为 prediction_result.png')

        # 【修复3】释放GPU显存缓存（防止100轮后显存碎片化）
        del a, b, l, pred
        torch.cuda.empty_cache()
    net.eval()

    with torch.no_grad():
        test_correct, test_total = 0, 0
        test_iou_sum, test_iou_count = 0, 0
        for i ,(a,b,l) in enumerate(test_loader):
            a,b,l = a.to(device),b.to(device),l.to(device)
            pred = net(a,b)
            pred_binary = (pred > 0.5).float()
            test_correct += (pred_binary==l).sum().item()
            test_total += l.numel()

            # 计算每个样本的IoU（只看变化类）
            intersection = (pred_binary * l).sum(dim=(1,2,3))
            union = ((pred_binary + l) > 0).float().sum(dim=(1,2,3))
            batch_iou = (intersection / (union + 1e-6)).mean().item()
            test_iou_sum += batch_iou
            test_iou_count += 1

        avg_iou = test_iou_sum / max(test_iou_count, 1)
        print("测试完成！")
    print(f'epoch :{epoch +1}'
          f'  loss {train_loss/train_total:.4f}'
          f'  acc {train_correct/train_total:.3f}'
          f'  test_acc {test_correct/test_total:.3f}'
          f'  IoU {avg_iou:.3f}')
    scheduler.step()  # 更新学习率

    #     for X, y in test_loader:
    #         X, y = X.to(device), y.to(device)
    #         y_pred = net(X)
    #         train_correct += (y_pred.argmax(1)==y).sum().item()
    #         test_total += y.numel()

