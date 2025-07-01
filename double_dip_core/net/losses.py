import numpy as np
import torch
from torch import nn
from torch.nn import functional

from double_dip_core.net.layers import GrayscaleLayer


class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(
            data=torch.tensor(blur, dtype=torch.get_default_dtype(), device=torch.get_default_device()),
            requires_grad=False)

        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(
            data=torch.tensor(image, dtype=torch.get_default_dtype(), device=torch.get_default_device()),
            requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))


class ExclusionLoss(nn.Module):
    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2).to(dtype=torch.get_default_dtype(),
                                                           device=torch.get_default_device())
        self.sigmoid = nn.Sigmoid().to(dtype=torch.get_default_dtype(), device=torch.get_default_device())

    def get_gradients(self, img1, img2):
        gradx_loss = []
        grady_loss = []

        for l in range(self.level):
            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)
            alphay = 1
            alphax = 1
            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)
            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)
        return gradx_loss, grady_loss

    def _all_comb(self, grad1_s, grad2_s):
        v = []
        for i in range(3):
            for j in range(3):
                v.append(torch.mean(((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2))) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss = self.get_gradients(img1, img2)
        loss_gradxy = sum(gradx_loss) / (self.level * 9) + sum(grady_loss) / (self.level * 9)
        return loss_gradxy / 2.0

    def compute_gradient(self, img):
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady


class ExtendedL1Loss(nn.Module):
    """
    Also pays attention to the mask, to be relative to its size
    """

    def __init__(self):
        super(ExtendedL1Loss, self).__init__()
        self.l1 = nn.L1Loss().to(device=torch.get_default_device())

    def forward(self, a, b, mask):
        normalizer = self.l1(mask, torch.zeros(mask.shape).to(device=torch.get_default_device()))
        c = self.l1(mask * a, mask * b) / normalizer
        return c


class GrayLoss(nn.Module):
    """
    Segmentation mask regularization loss
    """

    def __init__(self):
        super(GrayLoss, self).__init__()
        self.l1 = nn.L1Loss().to(device=torch.get_default_device())

    def forward(self, x):
        y = torch.ones_like(x) / 2.
        return 1 / self.l1(x, y)
