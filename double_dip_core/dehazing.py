import sys
import time
from collections import namedtuple

import torch.nn as nn
from cv2.ximgproc import guidedFilter
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity

from double_dip_core.net import *
from double_dip_core.net.losses import StdLoss
from double_dip_core.net.noise import get_noise
from double_dip_core.utils.image_io import *
from double_dip_core.utils.imresize import np_imresize


def get_dark_channel(image, w=15):
    """
    Get the dark channel prior in the (RGB) image data.
    Args:
        image:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
            M is the height, N is the width, 3 represents R/G/B channels.
        w:  window size
    Return:
        An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = image.shape
    padded = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(image, p=0.0001, w=15):
    """Get the atmosphere light in the (RGB) image data.
    Args:
        image:      the 3 * M * N RGB image data ([0, L-1]) as numpy array
        w:      window for dark channel
        p:      percentage of pixels for estimating the atmosphere light
    Return:
        A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    image = image.transpose(1, 2, 0)
    # reference CVPR09, 4.4
    darkch = get_dark_channel(image, w)
    M, N = darkch.shape
    flatI = image.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:int(M * N * p)]  # find top M * N * p indexes
    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)


DehazeResult = namedtuple("DehazeResult", ['learned', 't', 'a', 'ssim', 'psnr'])


class DehazeImage(object):
    def __init__(self, image_name, image, num_iter=8000, plot_during_training=True,
                 show_every=500, gt_ambient=None, use_deep_channel_prior=True, clip=True, ssims=None, psnrs=None):
        """
        Initializes the Dehaze class with the given parameters.
        Args:
            image_name (str): Name of the image file.
            image (np.array C x H x W [0..1]): Input image affected by haze.
            num_iter (int): Number of optimization iterations.
            plot_during_training (bool): Whether to plot intermediate results.
            show_every (int): Frequency of displaying results.
            gt_ambient (np.array C x H x W [0..1]): Ground truth ambient light (if available).
            clip (bool): Whether to clip output values.
        """

        self.image_name = image_name
        self.image = image
        self.num_iter = num_iter
        self.plot_during_training = plot_during_training
        self.show_every = show_every
        self.ssims = [] if not ssims else ssims
        self.psnrs = [] if not psnrs else psnrs
        self.use_deep_channel_prior = use_deep_channel_prior
        self.gt_ambient = gt_ambient
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = 0.001
        self.parameters = None
        self.current_result = None
        self.clip = clip
        self.blur_loss = None
        self.best_result = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.ambient_out = None
        self.total_loss = None
        self.input_depth = 8
        self.post = None
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()
        self._init_all()

    def _init_all(self):
        """
        Calls all initialization functions in the correct order.
        """
        self._init_images()
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def _init_images(self):
        """
        Prepares the input image for processing
        """
        self.original_image = self.image.copy()
        factor = 1
        image = self.image
        while image.shape[1] >= 800 or image.shape[2] >= 800:
            new_shape_x, new_shape_y = self.image.shape[1] / factor, self.image.shape[2] / factor
            new_shape_x -= (new_shape_x % 32)
            new_shape_y -= (new_shape_y % 32)
            image = np_imresize(self.image, output_shape=(new_shape_x, new_shape_y))
            factor += 1

        self.images_torch = np_to_torch(self.image).to(dtype=self.dtype, device=self.device)

    def _init_nets(self):
        """
        Initializes the neural networks for dehazing:
        - image_net: Predicts the dehazed image.
        - mask_net: Predicts the transmission map.
        """
        input_depth = self.input_depth
        pad = 'reflection'

        image_net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.image_net = image_net.to(dtype=self.dtype, device=self.device)

        mask_net = skip(
            input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.to(dtype=self.dtype, device=self.device)

    def _init_inputs(self):
        """
        Generates input noise maps for the networks.
        """

        original_noise = torch_to_np(get_noise(self.input_depth, 'noise',
                                               (self.image.shape[1], self.image.shape[2]),
                                               var=1 / 10.).to(dtype=self.dtype,
                                                               device=self.device)
                                     .detach())
        self.image_net_inputs = np_to_torch(original_noise).to(dtype=self.dtype, device=self.device).detach()

        original_noise = torch_to_np(get_noise(self.input_depth, 'noise',
                                               (self.image.shape[1], self.image.shape[2]),
                                               var=1 / 10.).to(dtype=self.dtype,
                                                               device=self.device).detach())
        self.mask_net_inputs = np_to_torch(original_noise).to(dtype=self.dtype, device=self.device).detach()

        if self._is_learning_ambient():
            self.ambient_net_input = get_noise(self.input_depth, 'meshgrid',
                                               (self.image.shape[1], self.image.shape[2])
                                               ).to(dtype=self.dtype,
                                                    device=self.device).detach()

    def _init_ambient(self):
        """
        Initializes the ambient light estimation:
        - If learning ambient light, initializes a neural network.
        - Otherwise, uses ground truth or estimates it.
        """
        if self._is_learning_ambient():
            ambient_net = skip(
                self.input_depth, 3,
                num_channels_down=[8, 16, 32, 64, 128],
                num_channels_up=[8, 16, 32, 64, 128],
                num_channels_skip=[0, 0, 0, 4, 4],
                upsample_mode='bilinear',
                filter_size_down=3,
                filter_size_up=3,
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
            self.ambient_net = ambient_net.to(dtype=self.dtype, device=self.device)

        if isinstance(self.gt_ambient, np.ndarray):
            atmosphere = self.gt_ambient
        else:
            atmosphere = get_atmosphere(self.image)

        self.ambient_val = nn.Parameter(torch.tensor(atmosphere.reshape((1, 3, 1, 1)),
                                                     dtype=self.dtype,
                                                     device=self.device),
                                        requires_grad=False)

    def _init_parameters(self):
        """
        Collects trainable parameters from all relevant networks.
        """
        parameters = [p for p in self.image_net.parameters()] + \
                     [p for p in self.mask_net.parameters()]
        if self._is_learning_ambient():
            parameters += [p for p in self.ambient_net.parameters()]

        self.parameters = parameters

    def _init_loss(self):
        """
        Initializes loss functions for optimization. MSE for the image-image reconstruction and
        StdLoss to keep smooth the blur (explained in the paper)
        """

        self.mse_loss = torch.nn.MSELoss().to(dtype=self.dtype, device=self.device)
        self.blur_loss = StdLoss().to(dtype=self.dtype, device=self.device)

    def _is_learning_ambient(self):
        """
        Determines whether the ambient light is being learned during optimization.
        Return:
            Boolean: True if ambient light needs to be learned, False otherwise.
        """
        return not self.use_deep_channel_prior

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
            self._optimization_closure(j)  # Compute loss
            self._obtain_current_result(j)  # Evaluate current result

            if self.plot_during_training:
                self._plot_closure(j)

            optimizer.step()  # Update parameters

    def _optimization_closure(self, iteration):
        """
        Computes the loss and backpropagates the gradients.
        Args:
            iteration (int): iteration number
        """
        if iteration == self.num_iter - 1:
            reg_std = 0
        else:
            reg_std = 1 / 30.

        # Add noise to input image
        image_net_input = self.image_net_inputs + (self.image_net_inputs.clone().normal_() * reg_std)
        self.image_out = self.image_net(image_net_input)

        # Compute atmospheric light
        if isinstance(self.ambient_net, nn.Module):
            ambient_net_input = self.ambient_net_input + (self.ambient_net_input.clone().normal_() * reg_std)

            self.ambient_out = self.ambient_net(ambient_net_input)
        else:
            self.ambient_out = self.ambient_val

        # Compute transmission map
        self.mask_out = self.mask_net(self.mask_net_inputs)

        # Compute total loss
        self.blur_out = self.blur_loss(self.mask_out)
        self.total_loss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                        self.images_torch) + 0.005 * self.blur_out

        # Regularization for atmospheric light learning
        if self._is_learning_ambient():
            self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
            if iteration < 1000:
                self.total_loss += self.mse_loss(self.ambient_out, self.ambient_val * torch.ones_like(self.ambient_out))

        # Backpropagate gradients
        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, iteration):
        """
        Evaluates the current dehazed result and updates the best result using ssim metric
        Args:
            iteration (int): iteration number
        """
        if iteration == 0 or iteration % 2 == 1 or iteration == self.num_iter - 1:
            image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
            mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
            ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)

            reconstruc_image = mask_out_np * image_out_np + (1 - mask_out_np) * ambient_out_np
            ssim = structural_similarity(self.image.astype(image_out_np.dtype),
                                         reconstruc_image,
                                         channel_axis=0, data_range=1.0)
            psnr = compute_psnr(self.image.astype(image_out_np.dtype), reconstruc_image, data_range=1.0)

            self.psnrs.append(psnr)
            self.ssims.append(ssim)
            self.current_result = DehazeResult(learned=image_out_np, t=mask_out_np, a=ambient_out_np, ssim=ssim,
                                               psnr=psnr)

            # Update best result if ssim improves
            if self.best_result is None or self.best_result.ssim < self.current_result.ssim:
                self.best_result = self.current_result

    def _plot_closure(self, iteration):
        """
        Displays training progress by printing loss and saving intermediate images_remove.
        Args:
            iteration (int): iteration number
        """

        sys.stdout.write(f'\rIteration {iteration + 1}, '
                         f'Loss: {self.total_loss.item():.5f}, '
                         f'SSIM:{self.current_result.ssim:.3f}, '
                         f'PSNR: {self.current_result.psnr:.3f}'
                         )
        sys.stdout.flush()
        # Save current
        if iteration % self.show_every == self.show_every - 1:
            # current_result = self.current_result.t * self.current_result.learned + (
            #        1 - self.current_result.t) * self.current_result.a
            # save_image(self.image_name + "reconstruct_step_{}".format(iteration), current_result)
            save_image(self.image_name + "_step_{}_J".format(iteration), self.current_result.learned)
            save_image(self.image_name + "_step_{}_t".format(iteration), self.current_result.t)

            if self._is_learning_ambient():
                save_image(self.image_name + "_step_{}_a".format(iteration), self.current_result.a)

    def finalize(self):
        """
        Finalizes the dehazing process and saves the results.
        """
        self.final_image = np_imresize(self.best_result.learned, output_shape=self.original_image.shape[1:])
        self.final_t_map = np_imresize(self.best_result.t, output_shape=self.original_image.shape[1:])
        self.final_a = np_imresize(self.best_result.a, output_shape=self.original_image.shape[1:])

        # Refine transmission map
        mask_out_np = self.t_matting(self.final_t_map)  # Clipped inside
        # Compute final haze-free image
        # original_image = t*image + (1-t)*A
        # image = (original_image - (1 - t) * A) * (1/t)
        self.post = np.clip((self.original_image - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)

        # Save images_remove in output dir
        if len(self.ssims) == (1 + self.num_iter // 2 + (1 if self.num_iter % 2 == 1 else 0)) * 2:
            save_graph(self.image_name.split('_')[0] + "_ssim", self.ssims, self.num_iter * 2, title="SSIM")
            save_graph(self.image_name.split('_')[0] + "_psnr", self.psnrs, self.num_iter * 2, title="PSNR")

        save_image(self.image_name + "_original", self.original_image)
        save_image(self.image_name + "_t_final", mask_out_np)
        save_image(self.image_name + "_isolate_J", self.post)
        save_image(self.image_name + "_a_final", np.clip(self.final_a, 0, 1))
        save_image(self.image_name + "_learned_J", np.clip(self.final_image, 0, 1))

        plot_image_grid(self.image_name + "_a_and_t",
                        [self.best_result.a * np.ones_like(self.best_result.learned), self.best_result.t])

    def t_matting(self, mask_out_np):
        """
        Refines the transmission map using guided filtering.

        - Uses guided filtering to smooth the transmission map while preserving edges.
        - Clips values to ensure they remain within valid range.
        """
        refine_t = guidedFilter(self.original_image.transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


class DehazeVideo(object):
    def __init__(self, video_name, video_frames, fps, num_iter=3000, plot_during_training=True,
                 show_every=500, gt_ambients=None, use_deep_channel_prior=True, clip=True, save_frames=False,
                 ssims=None, psnrs=None):
        """
        Initializes the Dehaze class with the given parameters for multiple images_remove.
        Args:
            image_names (list): List of names for the image files.
            images_remove (list of np.array): List of images_remove affected by haze.
            num_iter (int): Number of optimization iterations.
            plot_during_training (bool): Whether to plot intermediate results.
            show_every (int): Frequency of displaying results.
            gt_ambient (np.array): Ground truth ambient light (if available).
            clip (bool): Whether to clip output values.
        """
        self.video_name = video_name
        self.images = video_frames
        self.fps = int(fps)
        self.num_iter = num_iter
        self.plot_during_training = plot_during_training
        self.save_frames = save_frames
        self.show_every = show_every
        self.ssims = [] if not ssims else ssims
        self.psnrs = [] if not psnrs else psnrs
        self.use_deep_channel_prior = use_deep_channel_prior
        self.gt_ambients = [None] * len(self.images) if gt_ambients is None else gt_ambients
        self.ambient_net = None
        self.image_net = None
        self.mask_net = None
        self.ambient_val = None
        self.mse_loss = None
        self.learning_rate = 0.001
        self.parameters = None
        self.current_result = None
        self.clip = clip
        self.blur_loss = None
        self.best_result = None
        self.best_results = [None] * len(self.images)
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.done = False
        self.limit = None
        self.ambient_out = None
        self.total_loss = None
        self.input_depth = 8
        self.post = [None] * len(self.images)
        self.device = torch.get_default_device()
        self.dtype = torch.get_default_dtype()
        self._init_all()

    def _init_all(self):
        """
        Calls all initialization functions in the correct order.
        """
        self._init_nets()
        self._init_ambient()
        self._init_inputs()
        self._init_parameters()
        self._init_loss()

    def _init_image(self, img_index):
        self.image_torch = np_to_torch(self.images[img_index]).to(dtype=self.dtype, device=self.device)

    def _init_nets(self):
        """
        Initializes the neural networks for dehazing:
        - image_net: Predicts the dehazed image.
        - mask_net: Predicts the transmission map.
        """
        input_depth = self.input_depth
        pad = 'reflection'

        image_net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.image_net = image_net.to(dtype=self.dtype, device=self.device)

        mask_net = skip(
            input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.to(dtype=self.dtype, device=self.device)

    def _init_ambient(self):
        """
        Initializes the ambient light estimation:
        - If learning ambient light, initializes a neural network.
        - Otherwise, uses ground truth or estimates it.
        """
        # True when there's gt_ambient
        if self._is_learning_ambient():
            ambient_net = skip(
                self.input_depth, 3,
                num_channels_down=[8, 16, 32, 64, 128],
                num_channels_up=[8, 16, 32, 64, 128],
                num_channels_skip=[0, 0, 0, 4, 4],
                upsample_mode='bilinear',
                filter_size_down=3,
                filter_size_up=3,
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
            self.ambient_net = ambient_net.to(dtype=self.dtype, device=self.device)

        if isinstance(self.gt_ambients[0], np.ndarray):
            atmospheres = self.gt_ambients
        else:
            atmospheres = []
            for i in range(0, len(self.images), self.fps):
                atmosphere = get_atmosphere(self.images[i])
                atmospheres.append(atmosphere)

        self.ambient_val = [nn.Parameter(torch.tensor(atmosphere.reshape((1, 3, 1, 1)),
                                                      dtype=self.dtype,
                                                      device=self.device),
                                         requires_grad=False) for atmosphere in atmospheres]

    def _init_inputs(self):
        """
        Generates input noise maps for the networks.
        """
        original_noises = create_augmentations(torch_to_np(get_noise(self.input_depth, 'noise',
                                                                     (self.images[0].shape[1], self.images[0].shape[2]),
                                                                     var=1 / 10.).to(dtype=self.dtype,
                                                                                     device=self.device)
                                                           .detach()))

        self.image_net_inputs = [
            np_to_torch(original_noise).to(dtype=self.dtype, device=self.device)
            .detach()
            for original_noise in original_noises]

        self.mask_net_inputs = [
            np_to_torch(original_noise).to(dtype=self.dtype, device=self.device)
            .detach()
            for original_noise in original_noises]
        if self._is_learning_ambient():
            self.ambient_net_input = get_noise(self.input_depth, 'meshgrid',
                                               (self.images[0].shape[1], self.images[0].shape[2])
                                               ).to(dtype=self.dtype,
                                                    device=self.device).detach()

    def _init_parameters(self):
        """
        Collects trainable parameters from all relevant networks.
        """
        parameters = [p for p in self.image_net.parameters()] + \
                     [p for p in self.mask_net.parameters()]
        if self._is_learning_ambient():
            parameters += [p for p in self.ambient_net.parameters()]

        self.parameters = parameters

    def _init_loss(self):
        """
        Initializes loss functions for optimization. MSE for the image-image reconstruction and
        StdLoss to keep smooth the blur (explained in the paper)
        """
        self.mse_loss = torch.nn.MSELoss().to(dtype=self.dtype, device=self.device)
        self.blur_loss = StdLoss().to(dtype=self.dtype, device=self.device)

    def _is_learning_ambient(self):
        """
        Determines whether the ambient light is being learned during optimization.
        Return:
            Boolean: True if ambient light needs to be learned, False otherwise.
        """
        return not self.use_deep_channel_prior

    def optimize(self):
        """
        Performs the optimization process for all images_remove.
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for img_index in range(len(self.images)):
            self._init_image(img_index)
            self.done = False
            self.best_result = None
            for j in range(self.num_iter):
                optimizer.zero_grad()
                self._optimization_closure(j, img_index)  # Compute loss for each image
                self._obtain_current_result(j, img_index)  # Evaluate current result for each image
                if self.plot_during_training:
                    self._plot_closure(j, img_index)
                if self.done:
                    break
                optimizer.step()  # Update parameters after processing all images_remove
            if not self.limit:
                self.limit = self.best_result.ssim

    def _optimization_closure(self, iteration, img_index):
        """
        Computes the loss and backpropagates the gradients.
        Args:
            iteration (int): iteration number
        """
        if iteration == self.num_iter - 1:
            reg_std = 0
        else:
            reg_std = 1 / 30.

        # Add noise to input image
        image_net_input = self.image_net_inputs[0] + (self.image_net_inputs[0].clone().normal_() * reg_std)
        self.image_out = self.image_net(image_net_input)

        # Compute atmospheric light
        if isinstance(self.ambient_net, nn.Module):
            ambient_net_input = self.ambient_net_input + (self.ambient_net_input.clone().normal_() * reg_std)
            self.ambient_out = self.ambient_net(ambient_net_input)
        else:
            self.ambient_out = self.ambient_val[img_index // self.fps]

        # Compute transmission map
        self.mask_out = self.mask_net(self.mask_net_inputs[0])

        # Compute total loss
        self.blur_out = self.blur_loss(self.mask_out)
        self.total_loss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                        self.image_torch) + 0.005 * self.blur_out
        # Regularization for atmospheric light learning
        if self._is_learning_ambient():
            self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
            if iteration < 1000:
                self.total_loss += self.mse_loss(self.ambient_out,
                                                 self.ambient_val[img_index // self.fps] * torch.ones_like(
                                                     self.ambient_out))

        # Backpropagate gradients
        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, iteration, img_index):
        """
        Evaluates the current dehazed result and updates the best result using ssim metric
        Args:
            iteration (int): iteration number
        """
        if iteration == 0 or iteration % 2 == 1 or iteration == self.num_iter - 1:
            image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
            mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
            ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)

            recontruc = mask_out_np * image_out_np + (1 - mask_out_np) * ambient_out_np
            # Compute ssim for image quality evaluation
            ssim = structural_similarity(self.images[img_index].astype(image_out_np.dtype),
                                         recontruc,
                                         channel_axis=0, data_range=1.0)
            psnr = compute_psnr(self.images[img_index].astype(image_out_np.dtype), recontruc, data_range=1.0)

            if img_index == 0:
                self.ssims.append(ssim)
                self.psnrs.append(psnr)
            self.current_result = DehazeResult(learned=image_out_np, t=mask_out_np, a=ambient_out_np, ssim=ssim,
                                               psnr=psnr)

            # Update best result if ssim improves
            if self.best_result is None or self.best_result.ssim < self.current_result.ssim:
                self.best_result = self.current_result
                self.best_results[img_index] = self.best_result
                if self.limit and self.current_result.ssim > min(self.limit, 0.90):
                    self.done = True

    def _plot_closure(self, iteration, img_index):
        """
        Displays training progress by printing loss and saving intermediate images_remove.
        Args:
            iteration (int): iteration number
        """

        sys.stdout.write(f'\rFrame {img_index + 1}/{len(self.images)}, '
                         f'Iteration {iteration + 1}, '
                         f'Loss {self.total_loss.item():.5f}, '
                         f'SSIM:{self.current_result.ssim:.3f}, '
                         f'PSNR: {self.current_result.psnr:.3f}'
                         )
        sys.stdout.flush()
        # Save images_remove
        if img_index == 0 and iteration % self.show_every == self.show_every - 1:
            save_image(self.video_name + "_step_{}_J".format(iteration), self.current_result.learned)
            save_image(self.video_name + "_step_{}_t".format(iteration), self.current_result.t)
            if self._is_learning_ambient():
                save_image(self.video_name + "_frame_0_step_{}_a".format(iteration), self.current_result.a)

    def finalize(self):
        """
        Finalizes the dehazing process for all images_remove.
        """
        for i, image in enumerate(self.images):
            # self.image = image
            self.final_image = np_imresize(self.best_results[i].learned, output_shape=self.images[i].shape[1:])
            self.final_t_map = np_imresize(self.best_results[i].t, output_shape=self.images[i].shape[1:])
            self.final_a = np_imresize(self.best_results[i].a, output_shape=self.images[i].shape[1:])

            # Refine transmission map
            mask_out_np = self.t_matting(self.final_t_map, i)
            # Compute final haze-free image
            self.post[i] = np.clip((self.images[i] - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)
            # Save images_remove in output dir
            if self.save_frames or i == 0:
                save_image(f'{self.video_name}_frame_{i}_original', np.clip(self.images[i], 0, 1))
                save_image(f'{self.video_name}_frame_{i}_t_final', mask_out_np)
                save_image(f'{self.video_name}_frame_{i}_isolate_J', self.post[i])
                save_image(f'{self.video_name}_frame_{i}_a_final', np.clip(self.final_a, 0, 1))
                save_image(f'{self.video_name}_frame_{i}_learned_J', np.clip(self.final_image, 0, 1))

            if i == 0 and len(self.ssims) == (1 + self.num_iter // 2 + (1 if self.num_iter % 2 == 1 else 0)) * 2:
                save_graph(self.video_name.rsplit('_', 1)[0] + "_frame_0_ssim", self.ssims, self.num_iter * 2,
                           title="SSIM")
                save_graph(self.video_name.rsplit('_', 1)[0] + "_frame_0_psnr", self.psnrs, self.num_iter * 2,
                           title="PSNR")

    def t_matting(self, mask_out_np, img_index):
        """
        Refines the transmission map using guided filtering.

        - Uses guided filtering to smooth the transmission map while preserving edges.
        - Clips values to ensure they remain within valid range.
        """
        refine_t = guidedFilter(self.images[img_index].transpose(1, 2, 0).astype(np.float32),
                                mask_out_np[0].astype(np.float32), 50, 1e-4)
        if self.clip:
            return np.array([np.clip(refine_t, 0.1, 1)])
        else:
            return np.array([np.clip(refine_t, 0, 1)])


def main_dehaze_video(video_path, conf_params={}):
    """
    Performs haze removal (dehazing) on a video using a layer optimization
    method based on Deep Image Prior.

    Args:
        img_path (str): File path to the haze image
        conf_params (dic): Contains the key,value args for Dehaze class
    """
    # Prepara image format
    video_frames, fps = prepare_video(video_path)
    # Obtain the first approximation of the image
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    dh = DehazeVideo(video_name + "_0", video_frames, fps, **conf_params)
    dh.optimize()
    dh.finalize()

    # Obtain the optimum ambient lightness value
    if dh.use_deep_channel_prior:
        conf_params["gt_ambients"] = []
        conf_params["use_deep_channel_prior"] = False
        for best_result in dh.best_results:
            conf_params["gt_ambients"].append(best_result.a)

    # Upgrade the first obtained image
    dh = DehazeVideo(video_name + "_1", dh.post, fps, ssims=dh.ssims, psnrs=dh.psnrs, **conf_params)
    dh.optimize()
    dh.finalize()

    # Save it as a video
    dehaze_video_path = save_video(video_name, dh.post, fps)
    analyze_tssim_videos([video_path, dehaze_video_path])


def main_dehaze_image(img_path, conf_params={}):
    """
    Performs haze removal (dehazing) on an image using a layer optimization
    method based on Deep Image Prior.

    Args:
        img_path (str): File path to the haze image
        conf_params (dic): Contains the key,value args for Dehaze class
    """
    # Prepara image format
    image = prepare_image(img_path)
    image_name = os.path.splitext(os.path.basename(img_path))[0]

    # Obtain the first aproxtimation of the image
    dh = DehazeImage(image_name + "_0", image, **conf_params)
    dh.optimize()
    dh.finalize()

    # Obtain the optimum ambient lightness value
    if dh.use_deep_channel_prior:
        conf_params["gt_ambient"] = dh.best_result.a
        conf_params["use_deep_channel_prior"] = False

    # Upgrade the first obtained image
    dh = DehazeImage(image_name + "_1", dh.post, ssims=dh.ssims, psnrs=dh.psnrs, **conf_params)
    dh.optimize()
    dh.finalize()

    # Save original image
    save_image(image_name + "_original", np.clip(image, 0, 1))


if __name__ == "__main__":
    set_gpu_or_cpu_and_dtype(use_gpu=True, torch_dtype=torch.float32)

    conf_params = {
        "num_iter": 8000,
        "show_every": 500
    }
    start = time.time()
    main_dehaze_image("images/dehaze/hongkong.png", conf_params)
    print(f'\nTime: {time.time() - start} seconds')

    conf_params = {
        "num_iter": 4000,
        "show_every": 500
    }
    start = time.time()
    main_dehaze_video("videos/foggy_road.mp4", conf_params)
    print(f'\nTime: {time.time() - start} seconds')
