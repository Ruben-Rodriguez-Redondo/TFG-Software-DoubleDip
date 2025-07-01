import sys
import time
from collections import namedtuple

import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity

from double_dip_core.net import *
from double_dip_core.net.losses import ExtendedL1Loss
from double_dip_core.net.noise import get_noise
from double_dip_core.utils.image_io import *

WatermarkResult = namedtuple("WatermarkResult", ['clean', 'watermark', 'mask', 'ssim'])


class Watermark(object):
    def __init__(self, image_name, image, plot_during_training=True, show_every=500, num_iter_first_step=4000,
                 num_iter_second_step=7000,
                 watermark_hint=None):
        self.image = image
        self.image_name = image_name
        self.plot_during_training = plot_during_training
        self.show_every = show_every
        self.watermark_hint_torchs = None
        self.watermark_hint = watermark_hint
        self.ssims = []
        self.psnrs = []
        self.clean_net = None
        self.watermark_net = None
        self.image_torchs = None
        self.clean_net_inputs = None
        self.watermark_net_inputs = None
        self.clean_net_output = None
        self.watermark_net_output = None
        self.parameters = None
        self.num_iter_first_step = num_iter_first_step
        self.num_iter_second_step = num_iter_second_step
        self.input_depth = 2
        self.total_loss = None
        self.current_gradient = None
        self.current_result = None
        self.current_ssim = 0
        self.current_psnr = 0
        self.best_result = None
        self.learning_rate = 0.001
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()
        self._init_all()

    def _init_all(self):
        """
        Calls all initialization functions in the correct order.
        """
        self._init_images()
        self._init_losses()
        self._init_nets()
        self._init_parameters()
        self._init_noise()

    def _init_images(self):
        """
        Prepares the input image for processing
        """
        image_aug = create_augmentations(self.image)
        self.image_torchs = [np_to_torch(image).to(dtype=self.dtype, device=self.device)
                             for image in image_aug]
        water_mark_aug = create_augmentations(self.watermark_hint)
        self.watermark_hint_torchs = [
            np_to_torch(watr).to(dtype=self.dtype, device=self.device) for watr in
            water_mark_aug]

    def _init_losses(self):
        """
        Initializes loss functions for optimization.
        """
        self.l1_loss = nn.L1Loss().to(dtype=self.dtype, device=self.device)
        self.extended_l1_loss = ExtendedL1Loss().to(dtype=self.dtype, device=self.device)

    def _init_nets(self):
        """
        Initializes the neural networks for dehazing:
        - clean_net: Predicts the clean image.
        - mask_net: Predicts the mask.
        -watermark_net: Predict the watermark
        """
        pad = 'reflection'
        clean = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.clean_net = clean.to(dtype=self.dtype, device=self.device)

        watermark = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.watermark_net = watermark.to(dtype=self.dtype, device=self.device)

        mask = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64],
            num_channels_up=[8, 16, 32, 64],
            num_channels_skip=[0, 0, 0, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')
        self.mask_net = mask.to(dtype=self.dtype, device=self.device)

    def _init_parameters(self):
        """
        Collects trainable parameters from all relevant networks.
        """
        self.parameters = [p for p in self.clean_net.parameters()] + \
                          [p for p in self.watermark_net.parameters()] + \
                          [p for p in self.mask_net.parameters()]

    def _init_noise(self):
        """
        Initializes nets whith random noise
        """
        input_type = 'noise'
        dtype = self.dtype
        device = self.device
        clean_net_inputs = create_augmentations(torch_to_np(get_noise(self.input_depth, input_type,
                                                                      (self.image_torchs[0].shape[2],
                                                                       self.image_torchs[0].shape[3])).to(dtype=dtype,
                                                                                                          device=device).detach()))
        self.clean_net_inputs = [np_to_torch(clean_net_input).to(dtype=dtype, device=device).detach()
                                 for clean_net_input in clean_net_inputs]

        watermark_net_inputs = create_augmentations(torch_to_np(get_noise(self.input_depth, input_type,
                                                                          (self.image_torchs[0].shape[2],
                                                                           self.image_torchs[0].shape[3])).to(
            dtype=dtype, device=device).detach()))
        self.watermark_net_inputs = [np_to_torch(clean_net_input).to(dtype=dtype, device=device).detach()
                                     for clean_net_input in watermark_net_inputs]

        mask_net_inputs = create_augmentations(torch_to_np(get_noise(self.input_depth, input_type,
                                                                     (self.image_torchs[0].shape[2],
                                                                      self.image_torchs[0].shape[3])).to(dtype=dtype,
                                                                                                         device=device).detach()))
        self.mask_net_inputs = [np_to_torch(clean_net_input).to(dtype=dtype, device=device).detach()
                                for clean_net_input in mask_net_inputs]

    def optimize(self):
        """
        Performs the optimization process to dehaze the image.
        - Uses Adam optimizer with the specified learning rate.
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # step 1
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter_first_step):
            optimizer.zero_grad()
            self._step1_optimization_closure(j)
            optimizer.step()
            if self.plot_during_training:
                self._iteration_plot_closure(j, 1)
        # step 2
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iter_second_step):
            optimizer.zero_grad()
            self._step2_optimization_closure(j)
            self._step2_finalize_iteration(j)
            if self.plot_during_training:
                self._iteration_plot_closure(j, 2)
            optimizer.step()
        self._update_result_closure()
        self._second_step_plot_closure()

    def _step1_optimization_closure(self, iteration):
        """
        Computes the loss and backpropagates the gradients.
        Args:
            iteration (int): iteration number
        """
        if iteration == self.num_iter_first_step - 1:
            reg_noise_std = 0
            aug = 0
        else:
            reg_noise_std = (1 / 1000.) * min(iteration // (self.num_iter_first_step // 10), 10)
            aug = self._get_augmentation(iteration)

        clean_net_input = self.clean_net_inputs[aug] + (self.clean_net_inputs[aug].clone().normal_() * reg_noise_std)
        # applies the net
        self.clean_net_output = self.clean_net(clean_net_input)
        self.total_loss = 0
        self.total_loss += self.extended_l1_loss(self.clean_net_output,
                                                 self.image_torchs[aug],
                                                 (1 - self.watermark_hint_torchs[aug]))
        self.total_loss.backward(retain_graph=True)

    def _step2_optimization_closure(self, iteration):
        """
        Computes the loss and backpropagates the gradients.
        Args:
            step (int): iteration number
        """
        if iteration == self.num_iter_second_step - 1:
            reg_noise_std = 0
        else:
            reg_noise_std = (1 / 1000.) * min(iteration // (self.num_iter_second_step // 10), 10)

        aug = self._get_augmentation(iteration)
        if iteration == self.num_iter_second_step - 1:
            aug = 0
        # creates left_net_inputs and right_net_inputs by adding small noise
        clean_net_input = self.clean_net_inputs[aug] + (self.clean_net_inputs[aug].clone().normal_() * reg_noise_std)
        watermark_net_input = self.watermark_net_inputs[aug] + (
                self.watermark_net_inputs[aug].clone().normal_() * reg_noise_std)
        mask_net_input = self.mask_net_inputs[aug]
        # applies the nets
        self.clean_net_output = self.clean_net(clean_net_input)
        self.watermark_net_output = self.watermark_net(watermark_net_input)
        self.mask_net_output = self.mask_net(mask_net_input)
        self.total_loss = 0
        # loss on clean region
        self.total_loss += self.extended_l1_loss(self.clean_net_output,
                                                 self.image_torchs[aug],
                                                 (1 - self.watermark_hint_torchs[aug]))
        # loss in second region
        self.total_loss += 0.5 * self.l1_loss(self.watermark_hint_torchs[aug] *
                                              self.mask_net_output * self.watermark_net_output
                                              +
                                              (1 - self.mask_net_output) * self.clean_net_output,
                                              self.image_torchs[aug])  # this part learns the watermark
        self.total_loss.backward(retain_graph=True)

    def _get_augmentation(self, iteration):
        """
        Determines the augmentation type based on the iteration number.
        """
        if iteration % 4 in [1, 2, 3]:
            return 0
        iteration //= 2
        return iteration % 8

    def _step2_finalize_iteration(self, iteration):
        if iteration % 4 != 0 or iteration == self.num_iter_second_step - 1:
            clean_out_np = torch_to_np(self.clean_net_output)
            watermark_out_np = torch_to_np(self.watermark_net_output)
            mask_out_np = torch_to_np(self.mask_net_output)

            reconstruc = mask_out_np * self.watermark_hint * watermark_out_np + (1 - mask_out_np) * clean_out_np

            self.current_ssim = structural_similarity(self.image.astype(mask_out_np.dtype),
                                                      reconstruc, channel_axis=0, data_range=1.0)

            self.current_psnr = compute_psnr(self.image.astype(mask_out_np.dtype), reconstruc, data_range=1.0)

            self.ssims.append(self.current_ssim)
            self.psnrs.append(self.current_psnr)

    def _iteration_plot_closure(self, iteration, step):
        """
        Compute the curren_ssim and print the actual step
        """

        if iteration % self.show_every == self.show_every - 1:
            save_image(self.image_name + "_step_{}_clean_{}".format(step, iteration),
                       torch_to_np(self.clean_net_output))

            if step == 2:
                save_image(self.image_name + "_step_{}_watermark_{}".format(step, iteration),
                           torch_to_np(self.watermark_net_output))
                save_image(self.image_name + "_step_{}_mask_{}".format(step, iteration),
                           torch_to_np(self.mask_net_output))

        sys.stdout.write(f'\r Step {step}, Iteration {iteration + 1}, Loss: {self.total_loss.item():.5f}, '
                         f'SSIM: {self.current_ssim:.3f}, '
                         f'PSNR: {self.current_psnr:.3f}')

        sys.stdout.flush()

    def _update_result_closure(self):
        """
        Updates the best result find
        """
        self.current_result = WatermarkResult(clean=torch_to_np(self.clean_net_output),
                                              watermark=torch_to_np(self.watermark_net_output),
                                              mask=torch_to_np(self.mask_net_output),
                                              ssim=self.current_ssim)
        self.best_result = self.current_result

    def _second_step_plot_closure(self):
        """
        Save final best results at the end of the second step
        """
        step_number = 2
        if self.watermark_hint is not None:
            plot_image_grid("watermark_hint_and_mask_{}".format(step_number),
                            [np.clip(self.watermark_hint, 0, 1),
                             np.clip(torch_to_np(self.mask_net_output), 0, 1)])

        plot_image_grid("watermark_clean_{}".format(step_number),
                        [np.clip(torch_to_np(self.watermark_net_output), 0, 1),
                         np.clip(torch_to_np(self.clean_net_output), 0, 1)])

        plot_image_grid("learned_image_{}".format(step_number),
                        [np.clip(self.watermark_hint * torch_to_np(self.watermark_net_output) +
                                 torch_to_np(self.clean_net_output),
                                 0, 1), self.image])

    def finalize(self):
        """
        Saves the final results.
        """
        save_graph(self.image_name + "_ssim", self.ssims, self.num_iter_second_step, title="SSIM (Step 2)")
        save_graph(self.image_name + "_psnr", self.psnrs, self.num_iter_second_step, title="PSNR (Step 2)")

        save_image(self.image_name + "_watermark_finalize", self.best_result.watermark)
        save_image(self.image_name + "_clean_finalize", self.best_result.clean)
        save_image(self.image_name + "_mask_finalize", self.best_result.mask)
        save_image(self.image_name + "_final_finalize", (1 - self.watermark_hint) * self.image +
                   self.best_result.clean * self.watermark_hint)
        save_image(self.image_name + "_original", self.image)
        save_image(self.image_name + "_hint", self.watermark_hint)

        recovered_mask = self.watermark_hint * self.best_result.mask
        clear_image_places = np.zeros_like(recovered_mask)
        clear_image_places[recovered_mask < 0.1] = 1
        save_image(self.image_name + "_post_process_final", clear_image_places * self.image + (1 - clear_image_places) *
                   self.best_result.clean)
        recovered_watermark = self.watermark_hint * self.best_result.mask * self.best_result.watermark
        save_image(self.image_name + "_post_process_watermark", recovered_watermark)


ManyImageWatermarkResult = namedtuple("ManyImageWatermarkResult", ['cleans', 'mask', 'watermark', 'ssim', 'psnr'])


class ManyImagesWatermarkNoHint(object):
    def __init__(self, images_names, images, plot_during_training=True, show_every=500, num_iterations=4000
                 ):
        self.images = images
        self.images_names = images_names
        self.plot_during_training = plot_during_training
        self.show_every = show_every
        self.num_iterations = num_iterations
        self.ssims = [[] for _ in range(len(self.images))]
        self.psnrs = [[] for _ in range(len(self.images))]

        self.clean_nets = []
        self.watermark_net = None
        self.images_torch = None
        self.clean_nets_inputs = None
        self.clean_nets_outputs = None
        self.watermark_net_input = None
        self.watermark_net_output = None
        self.mask_net_input = None
        self.mask_net_output = None
        self.parameters = None
        self.input_depth = 2
        self.total_loss = None

        self.current_gradient = None
        self.current_result = None
        self.best_result = None
        self.learning_rate = 0.001
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()
        self._init_all()

    def _init_all(self):
        """
        Calls all initialization functions in the correct order.
        """
        self._init_images()
        self._init_losses()
        self._init_nets()
        self._init_parameters()
        self._init_noise()

    def _init_images(self):
        """
        Prepares the input image for processing
        """
        # convention - first dim is all the images_remove, second dim is the augmenations
        self.images_torch = [[np_to_torch(aug).to(dtype=self.dtype, device=self.device)
                              for aug in create_augmentations(image)] for image in self.images]

    def _init_losses(self):
        """
        Initializes loss functions for optimization.
        """
        self.l1_loss = nn.L1Loss().to(dtype=self.dtype, device=self.device)

    def _init_nets(self):
        """
        Initializes the neural networks for dehazing:
        - clean_net: Predicts clean images_remove.
        - mask_net: Predicts the mask .
        -right_net: Predicts the watermark
        """
        pad = 'reflection'
        cleans = [skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=5,
            filter_size_up=5,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU') for _ in self.images]

        self.clean_nets = [clean.to(dtype=self.dtype, device=self.device) for clean in
                           cleans]

        mask_net = skip(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.to(dtype=self.dtype, device=self.device)

        watermark = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.watermark_net = watermark.to(dtype=self.dtype, device=self.device)

    def _init_parameters(self):
        """
        Collects trainable parameters from all relevant networks.
        """
        self.parameters = sum([[p for p in clean_net.parameters()] for clean_net in self.clean_nets], []) + \
                          [p for p in self.mask_net.parameters()] + \
                          [p for p in self.watermark_net.parameters()]

    def _init_noise(self):
        """
        Initializes nets whith random noise
        """
        input_type = 'noise'
        dtype = self.dtype
        device = self.device
        self.clean_nets_inputs = []
        for image_idx in range(len(self.images)):
            original_noise = get_noise(self.input_depth, input_type,
                                       (self.images_torch[image_idx][0].shape[2],
                                        self.images_torch[image_idx][0].shape[3])).to(dtype=dtype,
                                                                                      device=device).detach()
            augmentations = create_augmentations(torch_to_np(original_noise))
            self.clean_nets_inputs.append(
                [np_to_torch(aug).to(dtype=dtype, device=device).detach() for aug in augmentations])

        original_noise = get_noise(self.input_depth, input_type,
                                   (self.images_torch[0][0].shape[2],
                                    self.images_torch[0][0].shape[3])).to(dtype=dtype, device=device).detach()
        augmentations = create_augmentations(torch_to_np(original_noise))
        self.mask_net_input = [np_to_torch(aug).to(dtype=dtype, device=device).detach() for aug in augmentations]

        original_noise = get_noise(self.input_depth, input_type,
                                   (self.images_torch[0][0].shape[2],
                                    self.images_torch[0][0].shape[3])).to(dtype=dtype, device=device).detach()
        augmentations = create_augmentations(torch_to_np(original_noise))
        self.watermark_net_input = [np_to_torch(aug).to(dtype=dtype, device=device).detach() for aug in augmentations]

    def optimize(self):
        """
        Performs the optimization process to dehaze the image.
        - Uses Adam optimizer with the specified learning rate.
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.num_iterations):
            optimizer.zero_grad()
            self._optimization_closure(j)
            self._finalize_iteration(j)
            if self.plot_during_training:
                self._iteration_plot_closure(j)
            optimizer.step()
        self._update_result_closure()
        self._step_plot_closure()

    def _update_result_closure(self):
        """
        Update best result
        """
        self.current_result = ManyImageWatermarkResult(cleans=[torch_to_np(c) for c in self.clean_nets_outputs],
                                                       watermark=torch_to_np(self.watermark_net_output),
                                                       mask=torch_to_np(self.mask_net_output),
                                                       ssim=self.current_ssim, psnr=self.current_psnr)
        self.best_result = self.current_result

    def _get_augmentation(self, iteration):
        """
        Determines the augmentation type based on the iteration number.
        """
        if iteration % 4 in [1, 2, 3]:
            return 0
        iteration //= 2
        return iteration % 8

    def _optimization_closure(self, iteration):
        """
        Computes the loss and backpropagates the gradients.
        Args:
            iteration (int): iteration number
        """
        aug = self._get_augmentation(iteration)

        if iteration == self.num_iterations - 1:
            reg_noise_std = 0
            aug = 0
        else:
            reg_noise_std = (1 / 1000.) * min(iteration // 100, 10)
        # adding small noise each iteration
        clean_nets_inputs = [clean_net_input[aug] + (clean_net_input[aug].clone().normal_() * reg_noise_std)
                             for clean_net_input in self.clean_nets_inputs]
        watermark_net_input = self.watermark_net_input[aug]
        mask_net_input = self.mask_net_input[aug]
        # applies the nets
        self.clean_nets_outputs = [clean_net(clean_net_input) for clean_net, clean_net_input
                                   in zip(self.clean_nets, clean_nets_inputs)]
        self.watermark_net_output = self.watermark_net(watermark_net_input)
        self.mask_net_output = self.mask_net(mask_net_input)
        self.total_loss = 0

        self.total_loss += sum(self.l1_loss(self.watermark_net_output * self.mask_net_output +
                                            clean_net_output * (1 - self.mask_net_output), image_torch[aug])
                               for clean_net_output, image_torch in zip(self.clean_nets_outputs, self.images_torch))
        self.total_loss.backward(retain_graph=True)

    def _finalize_iteration(self, iteration):
        if iteration == 0 or iteration % 4 != 0 or iteration == self.num_iterations - 1:
            clean_out_nps = [torch_to_np(clean_net_output) for clean_net_output in self.clean_nets_outputs]
            watermark_out_np = torch_to_np(self.watermark_net_output)
            mask_out_np = torch_to_np(self.mask_net_output)
            self.current_ssim = []
            self.current_psnr = []
            for i, clean_out_np in enumerate(clean_out_nps):
                reconstruc = clean_out_np * (1 - mask_out_np) + mask_out_np * watermark_out_np
                ssim = structural_similarity(self.images[i].astype(mask_out_np.dtype),
                                             reconstruc, channel_axis=0,
                                             data_range=1.0)
                psnr = compute_psnr(self.images[i].astype(mask_out_np.dtype),
                                    reconstruc, data_range=1.0)
                self.ssims[i].append(ssim)
                self.psnrs[i].append(psnr)
                self.current_ssim.append(ssim)
                self.current_psnr.append(psnr)

    def _iteration_plot_closure(self, iteration):
        """
        Displays training progress by printing loss and computes current_ssim
        Args:
            iteration (int): iteration number
        """
        sys.stdout.write(
            f'\r Iteration {iteration}, '
            f'Loss {self.total_loss.item():.5f}, '
            f'{", ".join([f"SSIM-{i + 1}: {v:.3f}" for i, v in enumerate(self.current_ssim)])}, '
            f'{", ".join([f"PSNR-{i + 1}: {v:.3f}" for i, v in enumerate(self.current_psnr)])}')
        sys.stdout.flush()
        if iteration % self.show_every == self.show_every - 1:
            clean_out_nps = [torch_to_np(clean_net_output) for clean_net_output in self.clean_nets_outputs]
            for name_img, clean_img in zip(self.images_names, clean_out_nps):
                save_image(name_img + "_clean_{}".format(iteration),
                           clean_img)

            save_image("watermark_{}".format(iteration),
                       torch_to_np(self.watermark_net_output))
            save_image("mask_{}".format(iteration),
                       torch_to_np(self.mask_net_output))

    def _step_plot_closure(self):
        """
        Saving final watermark_clean and learned_images in comparison with the original.
        """
        for image_name, image, clean_net_output in zip(self.images_names, self.images, self.clean_nets_outputs):
            plot_image_grid(image_name + "_watermark_clean",
                            [np.clip(torch_to_np(self.watermark_net_output), 0, 1),
                             np.clip(torch_to_np(clean_net_output), 0, 1)])
            plot_image_grid(image_name + "_learned_image",
                            [np.clip(torch_to_np(self.watermark_net_output) * torch_to_np(self.mask_net_output) +
                                     (1 - torch_to_np(self.mask_net_output)) * torch_to_np(clean_net_output),
                                     0, 1), image])

    def finalize(self):
        """
        Save results at the end of the process
        """

        for i, (image_name, clean, image, ssim_list, psnr_list) in enumerate(zip(
                self.images_names,
                self.best_result.cleans,
                self.images,
                self.ssims,
                self.psnrs)):
            save_graph(image_name + "_ssim", ssim_list, self.num_iterations, title="SSIM" + f"-{i + 1}")
            save_graph(image_name + "_psnr", psnr_list, self.num_iterations, title="PSNR" + f"-{i + 1}")

            save_image(image_name + "_clean_finalize", clean)
            save_image(image_name + "_original", image)

        save_image("watermark_finalize", self.best_result.watermark)
        save_image("mask_finalize", self.best_result.mask)
        obtained_watermark = self.best_result.mask * self.best_result.watermark

        obtained_imgs = self.best_result.cleans

        v = np.zeros_like(obtained_watermark)
        v[obtained_watermark < 0.1] = 1
        final_imgs = []
        # Upgrade the clean image
        for im, obt_im in zip(self.images, obtained_imgs):
            final_imgs.append(v * im + (1 - v) * obt_im)
        for img_name, final in zip(imgs_names, final_imgs):
            save_image(img_name + "_post_process_final", final)
        obtained_watermark[obtained_watermark < 0.1] = 0
        save_image("post_process_watermark", obtained_watermark)


def main_remove_watermark_hint(img_path, hint_path, conf_params={}):
    image = prepare_image(img_path)
    hint = prepare_image(hint_path)
    image_name = os.path.splitext(os.path.basename(img_path))[0]
    w = Watermark(image_name, image, watermark_hint=hint, **conf_params)
    w.optimize()
    w.finalize()


def main_remover_watermark_many_images(imgs_paths, imgs_names, conf_params={}):
    imgs = [prepare_image(img_path) for img_path in imgs_paths]
    for img_name, original in zip(imgs_names, imgs):
        save_image(img_name + "_original", original)

    w = ManyImagesWatermarkNoHint(imgs_names, imgs, plot_during_training=True, **conf_params)
    w.optimize()
    w.finalize()


if __name__ == "__main__":
    set_gpu_or_cpu_and_dtype(use_gpu=True, torch_dtype=torch.float32)

    imgs_paths = ['images/watermark/fotolia1.png', 'images/watermark/fotolia2.png', 'images/watermark/fotolia3.png']
    imgs_names = ['f1', 'f2', 'f3']
    conf_params = {
        "num_iterations": 4000,
        "show_every": 500
    }
    start = time.time()
    main_remover_watermark_many_images(imgs_paths, imgs_names, conf_params)
    print(f'\nTime: {time.time() - start} seconds')

    conf_params = {
        "num_iter_first_step": 4000,
        "num_iter_second_step": 7000,
        "show_every": 500
    }
    start = time.time()
    main_remove_watermark_hint('images/watermark/fotolia.png', 'images/watermark/fotolia_watermark.png',
                               conf_params=conf_params)
    print(f'\nTime: {time.time() - start} seconds')
