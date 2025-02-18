import sys
from collections import namedtuple

import torch.nn as nn
from cv2.ximgproc import guidedFilter
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

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


DehazeResult = namedtuple("DehazeResult", ['learned', 't', 'a', 'psnr'])


class DehazeVideo(object):
    def __init__(self, video_name, images, num_iter=3000, plot_during_training=True,
                 show_every=3000, gt_ambients=None, clip=True, save_frames=False):
        """
        Initializes the Dehaze class with the given parameters for multiple images.
        Args:
            image_names (list): List of names for the image files.
            images (list of np.array): List of images affected by haze.
            num_iter (int): Number of optimization iterations.
            plot_during_training (bool): Whether to plot intermediate results.
            show_every (int): Frequency of displaying results.
            gt_ambient (np.array): Ground truth ambient light (if available).
            clip (bool): Whether to clip output values.
        """
        self.video_name = video_name
        self.images = images
        self.num_iter = num_iter
        self.plot_during_training = plot_during_training
        self.save_frames = save_frames
        self.show_every = show_every
        self.use_deep_channel_prior = False if gt_ambients is not None else True
        self.gt_ambients = [None] * len(self.images)
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
        Prepares the input images for processing.
        """
        self.images_torch = [np_to_torch(image).type(torch.cuda.FloatTensor) for image in self.images]

    def _init_nets(self):
        """
        Initializes the neural networks for dehazing:
        - image_net: Predicts the dehazed image.
        - mask_net: Predicts the transmission map.
        """
        input_depth = self.input_depth
        data_type = torch.cuda.FloatTensor
        pad = 'reflection'

        image_net = skip(
            input_depth, 3,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.image_net = image_net.type(data_type)

        mask_net = skip(
            input_depth, 1,
            num_channels_down=[8, 16, 32, 64, 128],
            num_channels_up=[8, 16, 32, 64, 128],
            num_channels_skip=[0, 0, 0, 4, 4],
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.type(data_type)

    def _init_inputs(self):
        """
        Generates input noise maps for the networks.
        """
        original_noises = create_augmentations(torch_to_np(get_noise(self.input_depth, 'noise',
                                                                     (self.images[0].shape[1], self.images[0].shape[2]),
                                                                     var=1 / 10.).type(
            torch.cuda.FloatTensor).detach()))

        self.image_net_inputs = [np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()
                                 for original_noise in original_noises]

        self.mask_net_inputs = [np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()
                                for original_noise in original_noises]
        if self._is_learning_ambient():
            self.ambient_net_input = get_noise(self.input_depth, 'meshgrid',
                                               (self.images[0].shape[1], self.images[0].shape[2])
                                               ).type(torch.cuda.FloatTensor).detach()

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
            self.ambient_net = ambient_net.type(torch.cuda.FloatTensor)

    def _init_ambient_value(self, img_index):
        if isinstance(self.gt_ambients[img_index], np.ndarray):
            atmosphere = self.gt_ambients[img_index]
        else:
            atmosphere = get_atmosphere(self.images[img_index])
        self.ambient_val = nn.Parameter(torch.tensor(atmosphere.reshape((1, 3, 1, 1)),
                                                     dtype=torch.float32, device="cuda"),
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
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.blur_loss = StdLoss().type(data_type)

    def _is_learning_ambient(self):
        """
        Determines whether the ambient light is being learned during optimization.
        Return:
            Boolean: True if ambient light needs to be learned, False otherwise.
        """
        return not self.use_deep_channel_prior

    def optimize(self):
        """
        Performs the optimization process for all images.
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for img_index in range(len(self.images)):
            self._init_ambient_value(img_index)
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
                optimizer.step()  # Update parameters after processing all images
            if not self.limit:
                self.limit = self.best_result.psnr

    def _optimization_closure(self, iter, img_index):
        """
        Computes the loss and backpropagates the gradients.
        Args:
            iter (int): iteration number
        """
        if iter == self.num_iter - 1:
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
            self.ambient_out = self.ambient_val

        # Compute transmission map
        self.mask_out = self.mask_net(self.mask_net_inputs[0])

        # Compute total loss
        self.blur_out = self.blur_loss(self.mask_out)
        self.total_loss = self.mse_loss(self.mask_out * self.image_out + (1 - self.mask_out) * self.ambient_out,
                                        self.images_torch[img_index]) + 0.005 * self.blur_out

        # Regularization for atmospheric light learning
        if self._is_learning_ambient():
            self.total_loss += 0.1 * self.blur_loss(self.ambient_out)
            if iter < 1000:
                self.total_loss += self.mse_loss(self.ambient_out, self.ambient_val * torch.ones_like(self.ambient_out))

        # Backpropagate gradients
        self.total_loss.backward(retain_graph=True)

    def _obtain_current_result(self, iter, img_index):
        """
        Evaluates the current dehazed result and updates the best result using PSNR metric
        Args:
            iter (int): iteration number
        """
        if iter % 8 == 0:
            image_out_np = np.clip(torch_to_np(self.image_out), 0, 1)
            mask_out_np = np.clip(torch_to_np(self.mask_out), 0, 1)
            ambient_out_np = np.clip(torch_to_np(self.ambient_out), 0, 1)

            # Compute PSNR for image quality evaluation
            psnr = compare_psnr(self.images[img_index].astype(np.float32),
                                mask_out_np * image_out_np + (1 - mask_out_np) * ambient_out_np)
            self.current_result = DehazeResult(learned=image_out_np, t=mask_out_np, a=ambient_out_np, psnr=psnr)

            # Update best result if PSNR improves
            if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
                self.best_result = self.current_result
                self.best_results[img_index] = self.best_result
                if self.limit and self.current_result.psnr > self.limit:
                    self.done = True

    def _plot_closure(self, iter, img_index):
        """
        Displays training progress by printing loss and saving intermediate images.
        Args:
            iter (int): iteration number
        """

        sys.stdout.write(f'\rFrame {img_index:05d}   '
                         f'Iteration {iter:05d}    '
                         f'Loss {self.total_loss.item():.6f} '
                         f' current_psnr:{self.current_result.psnr:.6f} '
                         )
        sys.stdout.flush()
        # Save images
        if iter % self.show_every == self.show_every - 1:
            plot_image_grid(self.video_name + "_current_image_{}_{}".format(img_index, iter),
                            [self.images[img_index], np.clip(self.best_result.learned, 0, 1)])

    def finalize(self):
        """
        Finalizes the dehazing process for all images.
        """
        for i, image in enumerate(self.images):
            self.image = image
            self.final_image = np_imresize(self.best_results[i].learned, output_shape=self.images[i].shape[1:])
            self.final_t_map = np_imresize(self.best_results[i].t, output_shape=self.images[i].shape[1:])
            self.final_a = np_imresize(self.best_results[i].a, output_shape=self.images[i].shape[1:])

            # Refine transmission map
            mask_out_np = self.t_matting(self.final_t_map, i)
            # Compute final haze-free image
            self.post[i] = np.clip((self.images[i] - ((1 - mask_out_np) * self.final_a)) / mask_out_np, 0, 1)
            # Save images in output dir
            if self.save_frames:
                save_image(f'{self.video_name}_frame_{i}_original', np.clip(self.images[i], 0, 1))
                save_image(f'{self.video_name}_frame_{i}_t', mask_out_np)
                save_image(f'{self.video_name}_frame_{i}_final', self.post[i])
                save_image(f'{self.video_name}_frame_{i}_a', np.clip(self.final_a, 0, 1))

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


def main_dehaze(video_path, video_name, conf_params={}):
    """
    Performs haze removal (dehazing) on an image using a layer optimization
    method based on Deep Image Prior.

    Args:
        img_path (str): File path to the haze image
        conf_params (dic): Contains the key,value args for Dehaze class
    """
    # Prepara image format
    video_frames = prepare_video(video_path)
    # Obtain the first approximation of the image
    dh = DehazeVideo(video_name + "_{}".format(0), video_frames, **conf_params)
    dh.optimize()
    dh.finalize()

    # Obtain the optimum ambient lightness value
    if dh.use_deep_channel_prior:
        conf_params["gt_ambients"] = []
        for best_result in dh.best_results:
            conf_params["gt_ambients"].append(best_result.a)

    # Upgrade the first obtained image
    dh = DehazeVideo(video_name + "_{}".format(1), dh.post, **conf_params)
    dh.optimize()
    dh.finalize()

    # Save it as a video
    save_video(video_name, dh.post)


# todo check anohter metrics better than psnr
if __name__ == "__main__":
    conf_params = {
        "num_iter": 2000,
        "show_every": 3000
    }
    main_dehaze("videos/cut_haze_video.mp4", "haze", conf_params)
