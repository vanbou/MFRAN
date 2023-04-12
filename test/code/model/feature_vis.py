import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from model.mfran import MFRAN
from option import args



# 导入数据
def get_image_info(image_dir):
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    image_info = Image.open(image_dir).convert('RGB')
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)
    image_info = image_info.unsqueeze(0)
    import torch.nn as nn
    image_info = nn.Conv2d(3, 64, 3)(image_info)

    return image_info


# 获取第k层的特征图
def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index, layer in enumerate(feature_extractor):
            x = layer(x)
            if k == index:
                return x


#  可视化特征图
def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    save_path = "/home/server/wanbo/MSRN_2/src/results/"
    for index in range(1, feature_map_num + 1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1])
        plt.axis('off')
        scipy.misc.imsave(save_path + str(index) + ".png", feature_map[index - 1])
    plt.show()


if __name__ == '__main__':
    # 初始化图像的路径
    image_dir = r"/home/server/wanbo/MSRN_2/src/vis/set14/lenna.png"
    # 定义提取第几层的feature map
    k = 1
    # 导入Pytorch封装的AlexNet网络模型
    model = MFRAN(args)
    # model = models.alexnet(pretrained=True)
    # 是否使用gpu运算
    use_gpu = torch.cuda.is_available()
    use_gpu = False
    # 读取图像信息
    image_info = get_image_info(image_dir)
    # 判断是否使用gpu
    if use_gpu:
        model = model.cuda()
        image_info = image_info.cuda()
    # alexnet只有features部分有特征图
    # classifier部分的feature map是向量
    # feature_extractor = model.RDBs
    feature_extractor = model.body
    feature_map = get_k_layer_feature_map(feature_extractor, k, image_info)
    show_feature_map(feature_map)