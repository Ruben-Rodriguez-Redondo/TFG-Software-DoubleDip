import sys
from collections import namedtuple

import torch.nn as nn
from skimage.metrics import structural_similarity

from double_dip_core.net import skip, set_gpu_or_cpu_and_dtype
from double_dip_core.net.losses import ExclusionLoss
from double_dip_core.net.noise import get_noise
from double_dip_core.utils.image_io import *

TwoImagesSeparationResult = namedtuple("TwoImagesSeparationResult",
                                       ["reflection", "transmission", "ssim", "alpha1", "alpha2"])


class TwoImagesSeparation(object):
    def __init__(self, image1_name, image2_name, image1, image2, plot_during_training=True, show_every=500,
                 num_iter=4000,
                 original_reflection=None, original_transmission=None):
        # we assume the reflection is static
        self.image1 = image1
        self.image2 = image2
        self.plot_during_training = plot_during_training
        self.ssims = [[],[]]
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
        self.device =torch.get_default_device()
        self.dtype =torch.get_default_dtype()
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
        self.image1_torch = np_to_torch(self.image1).to(dtype=self.dtype,
                                                        device=self.device)
        self.image2_torch = np_to_torch(self.image2).to(dtype=self.dtype,
                                                        device=self.device)

    def _init_nets(self):
        """
        Initializes the neural networks for Separation:
        - reflection_net: Predicts the reflection layer.
        - transmission_net: Predicts the transmission layer.
        -alpha_net1: Mask 1
        -alpha_net2: Mask 2
        """
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

        self.reflection_net = reflection_net.to(dtype=self.dtype, device=self.device)

        transmission_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.to(dtype=self.dtype, device=self.device)
        alpha_net1 = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha1 = alpha_net1.to(dtype=self.dtype, device=self.device)

        alpha_net2 = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.alpha2 = alpha_net2.to(dtype=self.dtype, device=self.device)

    def _init_inputs(self):
        """
        Generates input noise maps for the networks.
        """
        input_type = 'noise'
        # input_type = 'meshgrid'
        dtype = self.dtype
        device = self.device
        self.reflection_net_input = get_noise(self.input_depth, input_type,
                                              (self.image1.shape[1], self.image1.shape[2])).to(dtype=dtype,
                                                                                               device=device).detach()
        self.alpha_net1_input = get_noise(self.input_depth, input_type,
                                          (self.image1.shape[1], self.image1.shape[2])).to(dtype=dtype,
                                                                                           device=device).detach()
        self.alpha_net2_input = get_noise(self.input_depth, input_type,
                                          (self.image1.shape[1], self.image1.shape[2])).to(dtype=dtype,
                                                                                           device=device).detach()
        self.transmission_net_input = get_noise(self.input_depth, input_type,
                                                (self.image1.shape[1], self.image1.shape[2])).to(dtype=dtype,
                                                                                                 device=device).detach()

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
        self.mse_loss = torch.nn.MSELoss().to(dtype=self.dtype, device=self.device)
        self.exclusion_loss = ExclusionLoss().to(dtype=self.dtype, device=self.device)

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

    def _optimization_closure(self, iteration):
        """
        Computes the loss and backpropagates the gradients.
        Args:
            iteration (int): iteration number
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

        if iteration < 1000:
            self.total_loss += 0.5 * self.mse_loss(self.current_alpha1,
                                                   torch.tensor([[[[0.5]]]]).to(dtype=self.dtype,
                                                                                device=self.device))
            self.total_loss += 0.5 * self.mse_loss(self.current_alpha2,
                                                   torch.tensor([[[[0.5]]]]).to(dtype=self.dtype,
                                                                                device=self.device))

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
        ssim1 = structural_similarity(self.image1.astype(v.dtype), v, channel_axis=0,
                                     data_range=1.0)
        ssim2 = structural_similarity(self.image2.astype(v.dtype), alpha2 * reflection_out_np + (1 - alpha2) * transmission_out_np, channel_axis=0,
                                     data_range=1.0)
        self.ssims[0].append(ssim1)
        self.ssims[1].append(ssim2)

        self.current_result = TwoImagesSeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                                        ssim=ssim1+ssim2, alpha1=alpha1, alpha2=alpha2)
        if self.best_result is None or self.best_result.ssim < self.current_result.ssim:
            self.best_result = self.current_result

    def _plot_closure(self, iteration):
        """
        Displays training progress by printing loss and saving intermediate images.
        Args:
            iteration (int): iteration number
        """

        sys.stdout.write(f'\rIteration {iteration} '
                         f'Loss {self.total_loss.item():.5f} '
                         f'Exclusion {self.exclusion.item():.5f} '
                         f'SSIM (SSIM1 + SSIM2): {self.current_result.ssim:.3f}')
        sys.stdout.flush()
        if iteration % self.show_every == self.show_every - 1:
            plot_image_grid("reflection_transmission_{}".format(iteration),
                            [self.current_result.reflection, self.current_result.transmission])
            save_image("sum1_{}".format(iteration), self.current_result.alpha1 * self.current_result.reflection +
                       (1 - self.current_result.alpha1) * self.current_result.transmission)
            save_image("sum2_{}".format(iteration), self.current_result.alpha2 * self.current_result.reflection +
                       (1 - self.current_result.alpha2) * self.current_result.transmission)

    def finalize(self):
        """
        Finalizes the separation process and saves the results.
        """
        save_graph(self.image1_name + "_ssim", self.ssims[0],self.num_iter) # Remove this or put it everywhere
        save_graph(self.image2_name + "_ssim", self.ssims[1],self.num_iter) # Remove this or put it everywhere

        save_image(self.image1_name + "_reflection", self.best_result.reflection)
        save_image(self.image1_name + "_transmission", self.best_result.transmission)
        save_image(self.image1_name + "_original", self.image1)
        save_image(self.image2_name + "_original", self.image2)


class Separation(object):
    def __init__(self, image_name, image, plot_during_training=True, show_every=500, num_iter=8000,
                 original_reflection=None, original_transmission=None):
        self.image = image
        self.plot_during_training = plot_during_training
        self.ssims = []
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
        self.device =torch.get_default_device()
        self.dtype =torch.get_default_dtype()
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
        self.images_torch = [np_to_torch(image).to(dtype=self.dtype, device=self.device)
                             for image in self.images]

    def _init_nets(self):
        """
        Initializes the neural networks for Transparency Separation:
        - reflection_net: Predicts the first layer.
        - transmision_net: Predicts the mask (no binary).
        """
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

        self.reflection_net = reflection_net.to(dtype=self.dtype, device=self.device)

        transmission_net = skip(
            self.input_depth, self.images[0].shape[0],
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.transmission_net = transmission_net.to(dtype=self.dtype, device=self.device)

    def _init_inputs(self):
        """
        Generates input noise maps for the networks.
        """
        input_type = 'noise'
        dtype = self.dtype
        device = self.device
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).to(dtype=dtype, device=device).detach())
        self.reflection_net_inputs = [np_to_torch(aug).to(dtype=dtype, device=device).detach() for aug in
                                      create_augmentations(origin_noise)]
        origin_noise = torch_to_np(get_noise(self.input_depth,
                                             input_type,
                                             (self.images_torch[0].shape[2],
                                              self.images_torch[0].shape[3])).to(dtype=dtype, device=device).detach())
        self.transmission_net_inputs = [np_to_torch(aug).to(dtype=dtype, device=device).detach() for aug in
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
        self.l1_loss = nn.L1Loss().to(dtype=self.dtype, device=self.device)
        self.exclusion_loss = ExclusionLoss().to(dtype=self.dtype, device=self.device)

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

    def _optimization_closure(self, iteration):
        """
        Computes the loss and backpropagates the gradients.
        Args:
            iteration (int): iteration number
        """
        if iteration == self.num_iter - 1:
            reg_noise_std = 0
        elif iteration < 1000:
            reg_noise_std = (1 / 1000.) * max((iteration // 100),1000)
        else:
            reg_noise_std = 1 / 1000.
        aug = self._get_augmentation(iteration)
        if iteration == self.num_iter - 1:
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

    def _obtain_current_result(self, iteration):
        """
        Also updates the best result
        Puts in self.current result the current result.
        """
        if iteration == 0 or iteration % 2 == 1 or iteration == self.num_iter - 1:
            reflection_out_np = np.clip(torch_to_np(self.reflection_out), 0, 1)
            transmission_out_np = np.clip(torch_to_np(self.transmission_out), 0, 1)
            ssim = structural_similarity(self.images[0].astype(reflection_out_np.dtype), reflection_out_np + transmission_out_np, channel_axis=0,
                                     data_range=1.0)
            self.ssims.append(ssim)
            self.current_result = SeparationResult(reflection=reflection_out_np, transmission=transmission_out_np,
                                                   ssim=ssim)
            if self.best_result is None or self.best_result.ssim < self.current_result.ssim:
                self.best_result = self.current_result

    def _plot_closure(self, iteration):
        """
        Displays training progress by printing loss and saving intermediate images.
        Args:
            iteration (int): iteration number
        """

        sys.stdout.write(
            f'\rIteration {iteration}    '
            f'Loss {self.total_loss.item():.5f}  '
            f'SSIM: {self.current_result.ssim:.3f}')
        sys.stdout.flush()
        if iteration % self.show_every == self.show_every - 1:
            plot_image_grid("left_right_{}".format(iteration),
                            [self.current_result.reflection, self.current_result.transmission])

    def finalize(self):
        """
        Finalizes the transparency separation process and saves the results.
        """
        save_graph(self.image_name + "_ssim", self.ssims,self.num_iter)
        save_image(self.image_name + "_reflection", self.best_result.reflection)
        save_image(self.image_name + "_transmission", self.best_result.transmission)
        save_image(self.image_name + "_reflection2", 2 * self.best_result.reflection)
        save_image(self.image_name + "_transmission2", 2 * self.best_result.transmission)
        save_image(self.image_name + "_original", self.images[0])


SeparationResult = namedtuple("SeparationResult", ['reflection', 'transmission', 'ssim'])


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
    set_gpu_or_cpu_and_dtype(use_gpu=True, torch_dtype=torch.float32)

    # Separation from two images
    conf_params = {
        "num_iter": 1000,
        "show_every":500
    }
    #main_two_images_separation('images/input1.jpg', 'images/input2.jpg', conf_params)

    # Separation of textures
    conf_params = {
        "num_iter": 4000,
        "show_every":500
    }
    main_separation('images/texture1.jpg', 'images/texture2.jpg', conf_params)
