import sys
from collections import namedtuple

import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from double_dip_core.net import skip
from double_dip_core.net.losses import ExclusionLoss
from double_dip_core.net.noise import get_noise
from double_dip_core.utils.image_io import *

TwoImagesSeparationResult = namedtuple("TwoImagesSeparationResult",
                                       ["reflection", "transmission", "psnr", "alpha1", "alpha2"])


class TwoImagesSeparation(object):
    def __init__(self, image1_name, image2_name, image1, image2, plot_during_training=True, show_every=500,
                 num_iter=4000,
                 original_reflection=None, original_transmission=None):
        # we assume the reflection is static
        self.image1 = image1
        self.image2 = image2
        self.plot_during_training = plot_during_training
        self.psnrs = []
        self.show_every = show_every
        self.image1_name = image1_name
        self.image2_name = image2_name
        self.num_iter = num_iter
        self.parameters = None
        self.learning_rate = 0.001
        self.input_depth = 2
        self.reflection_net_input = None
        self.transmission_net_input = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.total_loss = None
        self.multiscale_loss = None
        self.multiscale = None
        self.reflection_out = None
        self.transmission_out = None
        self.current_result = None
        self.best_result = None
        self._init_all()

    def _init_all(self):
        """
        Calls all initialization functions in the correct order.
        """
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        """
        Prepares the input image for processing
        """
        self.image1_torch = np_to_torch(self.image1).type(torch.cuda.FloatTensor)
        self.image2_torch = np_to_torch(self.image2).type(torch.cuda.FloatTensor)

    def _init_nets(self):
        """
        Initializes the neural networks for Separation:
        - reflection_net: Predicts the reflection layer.
        - transmission_net: Predicts the transmission layer.
        -alpha_net1: Mask 1
        -alpha_net2: Mask 2
        """
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)
        alpha_net1 = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha1 = alpha_net1.type(data_type)

        alpha_net2 = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha2 = alpha_net2.type(data_type)

    def _init_inputs(self):
        """
        Generates input noise maps for the networks.
        """
        input_type = 'noise'
        # input_type = 'meshgrid'
        data_type = torch.cuda.FloatTensor
        self.reflection_net_input = get_noise(self.input_depth, input_type,
                                              (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        self.alpha_net1_input = get_noise(self.input_depth, input_type,
                                          (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        self.alpha_net2_input = get_noise(self.input_depth, input_type,
                                          (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()
        self.transmission_net_input = get_noise(self.input_depth, input_type,
                                                (self.image1.shape[1], self.image1.shape[2])).type(data_type).detach()

    def _init_parameters(self):
        """
        Collects trainable parameters from all relevant networks.
        """
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]
        self.parameters += [p for p in self.alpha1.parameters()]
        self.parameters += [p for p in self.alpha2.parameters()]

    def _init_losses(self):
        """
        Initializes loss functions for optimization.
        """
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)

    def optimize(self):
        """
        Performs the optimization process to separate the images.
        - Uses Adam optimizer with the specified learning rate.
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result()
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def _optimization_closure(self, iter):
        """
        Computes the loss and backpropagates the gradients.
        Args:
            iter (int): iteration number
        """
        reg_noise_std = 0
        reflection_net_input = self.reflection_net_input + (self.reflection_net_input.clone().normal_() * reg_noise_std)
        transmission_net_input = self.transmission_net_input + \
                                 (self.transmission_net_input.clone().normal_() * reg_noise_std)

        self.reflection_out = self.reflection_net(reflection_net_input)
        self.transmission_out = self.transmission_net(transmission_net_input)
        alpha_net_input = self.alpha_net1_input + (self.alpha_net1_input.clone().normal_() * reg_noise_std)
        self.current_alpha1 = self.alpha1(alpha_net_input)[:, :,
                              self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                              self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1] * 0.9 + 0.05

        alpha_net_input = self.alpha_net2_input + (self.alpha_net2_input.clone().normal_() * reg_noise_std)
        self.current_alpha2 = self.alpha2(alpha_net_input)[:, :,
                              self.image1_torch.shape[2] // 2:self.image1_torch.shape[2] // 2 + 1,
                              self.image1_torch.shape[3] // 2:self.image1_torch.shape[3] // 2 + 1] * 0.9 + 0.05
        self.total_loss = self.mse_loss(self.current_alpha1 * self.reflection_out +
                                        (1 - self.current_alpha1) * self.transmission_out,
                                        self.image1_torch)
        self.total_loss += self.mse_loss(self.current_alpha2 * self.reflection_out +
                                         (1 - self.current_alpha2) * self.transmission_out,
                                         self.image2_torch)
        self.exclusion = self.exclusion_loss(self.reflection_out, self.transmission_out)
        self.total_loss += 0.1 * self.exclusion

        if iter < 1000:
            self.total_loss += 0.5 * self.mse_loss(self.current_alpha1,
                                                   torch.tensor([[[[0.5]]]]).type(torch.cuda.FloatTensor))
            self.total_loss += 0.5 * self.mse_loss(self.current_alpha2,
                                                   torch.tensor([[[[0.5]]]]).type(torch.cuda.FloatTensor))

        self.total_loss.backward()

    def _obtain_current_result(self):
        """
        Updates the best result.
        Puts in self.current result the current result.
        """
        reflection_out_np = np.clip(torch_to_np(self.reflection_out), 0, 1)
        transmission_out_np = np.clip(torch_to_np(self.transmission_out), 0, 1)
        alpha1 = np.clip(torch_to_np(self.current_alpha1), 0, 1)
        alpha2 = np.clip(torch_to_np(self.current_alpha2), 0, 1)
        v = alpha1 * reflection_out_np + (1 - alpha1) * transmission_out_np
        psnr1 = compare_psnr(self.image1, v)
        psnr2 = compare_psnr(self.image2, alpha2 * reflection_out_np + (1 - alpha2) * transmission_out_np)
        self.psnrs.append(psnr1 + psnr2)
        self.current_result = TwoImagesSeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                                        psnr=psnr1, alpha1=alpha1, alpha2=alpha2)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result

    def _plot_closure(self, iter):
        """
        Displays training progress by printing loss and saving intermediate images.
        Args:
            iter (int): iteration number
        """

        sys.stdout.write(f'\rIteration {iter:5d} '
                         f'Loss {self.total_loss.item():.5f} '
                         f'Exclusion {self.exclusion.item():.5f} '
                         f'PSNR_gt: {self.current_result.psnr:.6f}')
        sys.stdout.flush()
        if iter % self.show_every == self.show_every - 1:
            plot_image_grid("reflection_transmission_{}".format(iter),
                            [self.current_result.reflection, self.current_result.transmission])
            save_image("sum1_{}".format(iter), self.current_result.alpha1 * self.current_result.reflection +
                       (1 - self.current_result.alpha1) * self.current_result.transmission)
            save_image("sum2_{}".format(iter), self.current_result.alpha2 * self.current_result.reflection +
                       (1 - self.current_result.alpha2) * self.current_result.transmission)

    def finalize(self):
        """
        Finalizes the separation process and saves the results.
        """
        save_graph(self.image1_name + "_psnr", self.psnrs)
        save_image(self.image1_name + "_reflection", self.best_result.reflection)
        save_image(self.image1_name + "_transmission", self.best_result.transmission)
        save_image(self.image1_name + "_original", self.image1)
        save_image(self.image2_name + "_original", self.image2)


class Separation(object):
    def __init__(self, image_name, image, plot_during_training=True, show_every=500, num_iter=8000,
                 original_reflection=None, original_transmission=None):
        self.image = image
        self.plot_during_training = plot_during_training
        self.psnrs = []
        self.show_every = show_every
        self.image_name = image_name
        self.num_iter = num_iter
        self.parameters = None
        self.learning_rate = 0.0005
        self.input_depth = 3
        self.reflection_net_inputs = None
        self.transmission_net_inputs = None
        self.original_transmission = original_transmission
        self.original_reflection = original_reflection
        self.reflection_net = None
        self.transmission_net = None
        self.total_loss = None
        self.reflection_out = None
        self.transmission_out = None
        self.current_result = None
        self.best_result = None
        self._init_all()

    def _init_all(self):
        """
        Calls all initialization functions in the correct order.
        """
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_parameters()
        self._init_losses()

    def _init_images(self):
        """
        Prepares the input image for processing
        """
        self.images = create_augmentations(self.image)
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]

    def _init_nets(self):
        """
        Initializes the neural networks for Transparency Separation:
        - reflection_net: Predicts the first layer.
        - transmision_net: Predicts the mask (no binary).
        """
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'
        reflection_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.reflection_net = reflection_net.type(data_type)

        transmission_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.type(data_type)

    def _init_inputs(self):
        """
        Generates input noise maps for the networks.
        """
        input_type = 'noise'
        data_type = torch.cuda.FloatTensor
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).type(data_type).detach())
        self.reflection_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in
                                      create_augmentations(origin_noise)]
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).type(data_type).detach())
        self.transmission_net_inputs = [np_to_torch(aug).type(data_type).detach() for aug in
                                        create_augmentations(origin_noise)]

    def _init_parameters(self):
        """
        Collects trainable parameters from all relevant networks.
        """
        self.parameters = [p for p in self.reflection_net.parameters()] + \
                          [p for p in self.transmission_net.parameters()]

    def _init_losses(self):
        """
        Initializes loss functions for optimization.
        """
        data_type = torch.cuda.FloatTensor
        self.l1_loss = nn.L1Loss().type(data_type)
        self.exclusion_loss = ExclusionLoss().type(data_type)

    def optimize(self):
        """
        Performs the optimization process to dehaze the image.
        - Uses Adam optimizer with the specified learning rate.
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._obtain_current_result(j)
            if self.plot_during_training:
                self._plot_closure(j)
            optimizer.step()

    def _get_augmentation(self, iteration):
        """
        Determines the augmentation type based on the iteration number.
        """
        if iteration % 2 == 1:
            return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, iter):
        """
        Computes the loss and backpropagates the gradients.
        Args:
            iter (int): iteration number
        """
        if iter == self.num_iter - 1:
            reg_noise_std = 0
        elif iter < 1000:
            reg_noise_std = (1 / 1000.) * (iter // 100)
        else:
            reg_noise_std = 1 / 1000.
        aug = self._get_augmentation(iter)
        if iter == self.num_iter - 1:
            aug = 0
        reflection_net_input = self.reflection_net_inputs[aug] + (
                self.reflection_net_inputs[aug].clone().normal_() * reg_noise_std)
        transmission_net_input = self.transmission_net_inputs[aug] + (
                self.transmission_net_inputs[aug].clone().normal_() * reg_noise_std)

        self.reflection_out = self.reflection_net(reflection_net_input)

        self.transmission_out = self.transmission_net(transmission_net_input)

        self.total_loss = self.l1_loss(self.reflection_out + self.transmission_out, self.images_torch[aug])
        self.total_loss += 0.01 * self.exclusion_loss(self.reflection_out, self.transmission_out)
        self.total_loss.backward()

    def _obtain_current_result(self, iter):
        """
        Also updates the best result
        Puts in self.current result the current result.
        """
        if iter == self.num_iter - 1 or iter % 8 == 0:
            reflection_out_np = np.clip(torch_to_np(self.reflection_out), 0, 1)
            transmission_out_np = np.clip(torch_to_np(self.transmission_out), 0, 1)
            psnr = compare_psnr(self.images[0], reflection_out_np + transmission_out_np)
            self.psnrs.append(psnr)
            self.current_result = SeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                                   psnr=psnr)
            if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
                self.best_result = self.current_result

    def _plot_closure(self, iter):
        """
        Displays training progress by printing loss and saving intermediate images.
        Args:
            iter (int): iteration number
        """

        sys.stdout.write(
            f'\rIteration {iter:5d}    Loss {self.total_loss.item():.5f}  PSRN: {self.current_result.psnr:.6f}')
        sys.stdout.flush()
        if iter % self.show_every == self.show_every - 1:
            plot_image_grid("left_right_{}".format(iter),
                            [self.current_result.reflection, self.current_result.transmission])

    def finalize(self):
        """
        Finalizes the transparency separation process and saves the results.
        """
        save_graph(self.image_name + "_psnr", self.psnrs)
        save_image(self.image_name + "_reflection", self.best_result.reflection)
        save_image(self.image_name + "_transmission", self.best_result.transmission)
        save_image(self.image_name + "_reflection2", 2 * self.best_result.reflection)
        save_image(self.image_name + "_transmission2", 2 * self.best_result.transmission)
        save_image(self.image_name + "_original", self.images[0])


SeparationResult = namedtuple("SeparationResult", ['reflection', 'transmission', 'psnr'])


def main_two_images_separation(path_input1, path_input2, conf_params):
    input1 = prepare_image(path_input1)
    input2 = prepare_image(path_input2)
    input1_name = os.path.splitext(os.path.basename(path_input1))[0]  # "input1"
    input2_name = os.path.splitext(os.path.basename(path_input2))[0]
    t = TwoImagesSeparation(input1_name, input2_name, input1, input2, **conf_params)
    t.optimize()
    t.finalize()


def main_separation(path_input1, path_input2, conf_params):
    t1 = prepare_image(path_input1)
    t2 = prepare_image(path_input2)

    # Resize to its min if images have diferent size
    min_height = min(t1.shape[0], t2.shape[0])
    min_width = min(t1.shape[1], t2.shape[1])
    t1 = cv2.resize(t1, (min_width, min_height))
    t2 = cv2.resize(t2, (min_width, min_height))

    s = Separation('textures', (t1 + t2) / 2, **conf_params)
    s.optimize()
    s.finalize()


if __name__ == "__main__":
    # Separation from two images
    conf_params = {
        "num_iter": 1000
    }
    main_two_images_separation('images/input1.jpg', 'images/input2.jpg', conf_params)

    # Separation of textures
    conf_params = {
        "num_iter": 1000
    }
    main_separation('images/texture1.jpg', 'images/texture2.jpg', conf_params)
