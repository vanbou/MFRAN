import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from model.mfran import MFRAN
from option import args

def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')
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

def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index, layer in enumerate(feature_extractor):
            x = layer(x)
            if k == index:
                return x

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
    image_dir = r"/home/server/wanbo/MSRN_2/src/vis/set14/lenna.png"
    k = 1
    model = MFRAN(args)
    # model = models.alexnet(pretrained=True)
    use_gpu = torch.cuda.is_available()
    use_gpu = False
    image_info = get_image_info(image_dir)
    if use_gpu:
        model = model.cuda()
        image_info = image_info.cuda()
    feature_extractor = model.body
    feature_map = get_k_layer_feature_map(feature_extractor, k, image_info)
    show_feature_map(feature_map)