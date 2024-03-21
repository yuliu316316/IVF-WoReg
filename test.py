import os

import cv2
import numpy as np

from modules.module import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

filepath = './dataset/test'

scene_directory = [os.path.join(filepath, i) for i in os.listdir(filepath)]
filename = os.listdir(filepath)
filename1 = []

for i in filename:
    str_list = list(i)
    str_list.insert(-4, '_trans')
    str_out = ''.join(str_list)
    filename1.append(str_out)


def inference():
    model = Net().to(device)
    model.eval()
    model.load_state_dict(torch.load('./checkpoint/42.pkl'))

    # Load the image
    # Read Expo times in scene
    for i in range(len(scene_directory)):
        Img = cv2.imread(scene_directory[i])

        H, W = Img.shape[0], Img.shape[1] // 2
        ifuse_label_a, ifuse_label_b = Img[:, :W, :], Img[:, W:, :]
        H, W, _ = Img.shape
        H = int(H // 14 * 14)
        W = int(W // 2) // 14 * 14
        ifuse_label_a = cv2.resize(ifuse_label_a, dsize=(W, H))
        ifuse_label_b = cv2.resize(ifuse_label_b, dsize=(W, H))

        im1 = np.transpose(ifuse_label_a, (2, 0, 1))
        im1 = torch.from_numpy(im1).float() / 255.0
        im2 = np.transpose(ifuse_label_b, (2, 0, 1))
        im2 = torch.from_numpy(im2).float() / 255.0
        im1 = torch.unsqueeze(im1, 0).to(device)
        im2 = torch.unsqueeze(im2, 0).to(device)
        with torch.no_grad():
            pre, x1 = model(im1, im2)

        pre = torch.squeeze(pre, 0)
        pre = pre.cpu().numpy()  #
        pre = (pre - np.min(pre)) / (np.max(pre) - np.min(pre))
        pre = np.clip(pre * 255.0, 0., 255.)  #

        pre = pre.transpose(1, 2, 0)

        # cv2.imwrite('/home/l/data_1/YR/LJY/results/{}'.format(filename1[i]), pre)
        cv2.imwrite('./results/{}'.format(filename1[i]), pre)


if __name__ == '__main__':
    inference()
