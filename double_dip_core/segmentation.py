import sys
from collections import namedtuple

from cv2.ximgproc import guidedFilter
from skimage.metrics import structural_similarity

from double_dip_core.net import *
from double_dip_core.net.downsampler import *
from double_dip_core.net.losses import GradientLoss, ExtendedL1Loss, GrayLoss
from double_dip_core.net.noise import get_noise
from double_dip_core.utils.image_io import *


def otsu_intraclass_variance(image, threshold):
    """
    Compute otsu intra class variance given a threshold
    Args:
        image (numpy.ndarray): Grayscale image
        threshold (int): Segmentation threshold

    Returns:
        float: Intra-class variance
    """
    foreground = image >= threshold
    background = image < threshold

    w_fg = np.sum(foreground) / image.size
    w_bg = np.sum(background) / image.size

    var_fg = np.var(image[foreground]) if w_fg > 0 else 0
    var_bg = np.var(image[background]) if w_bg > 0 else 0

    return w_fg * var_fg + w_bg * var_bg


def obtain_bg_fg_segmentation_hints(image):
    s = image.copy()
    s = np.transpose(s, (1, 2, 0))

    # Convert image from [0, 1] to [0, 255]
    s = (s * 255).astype(np.uint8)
    s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)

    otsu_threshold = min(
        range(int(np.min(s)), (int(np.max(s)) + 1)),
        key=lambda th: otsu_intraclass_variance(s, th),
    )

    fg = s.copy()
    fg[s > otsu_threshold] = 255
    fg[s <= otsu_threshold] = 0

    bg = s.copy()
    bg[s > otsu_threshold] = 0
    bg[s <= otsu_threshold] = 255

    return fg, bg


SegmentationResult = namedtuple("SegmentationResult",
                                ['mask', 'learned_mask', 'left', 'right', 'ssim'])


class Segmentation(object):
    def __init__(self, image_name, image, plot_during_training=True,
                 first_step_iter_num=2000,
                 second_step_iter_num=4000,
                 bg_hint=None, fg_hint=None,
                 show_every=500,
                 downsampling_factor=0.1, downsampling_number=0):
        self.image = image
        self.fg_hint = fg_hint
        self.bg_hint = bg_hint

        if bg_hint is None or fg_hint is None:
            fg, bg = obtain_bg_fg_segmentation_hints(self.image)
            self.fg_hint = pil_to_np(fg)
            self.bg_hint = pil_to_np(bg)

        self.image_name = image_name
        self.plot_during_training = plot_during_training
        self.downsampling_factor = downsampling_factor
        self.downsampling_number = downsampling_number
        self.ssims = []
        self.mask_net = None
        self.show_every = show_every
        self.left_net = None
        self.right_net = None
        self.images = None
        self.images_torch = None
        self.left_net_inputs = None
        self.right_net_inputs = None
        self.mask_net_inputs = None
        self.left_net_outputs = None
        self.right_net_outputs = None
        self.second_step_done = False
        self.mask_net_outputs = None
        self.parameters = None
        self.fixed_masks = None
        self.first_step_iter_num = first_step_iter_num
        self.second_step_iter_num = second_step_iter_num
        self.input_depth = 2
        self.multiscale_loss = None
        self.total_loss = None
        self.current_gradient = None
        self.current_result = None
        self.best_result = None
        self.learning_rate = 0.001
        self.device =torch.get_default_device()
        self.dtype =torch.get_default_dtype()
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
        self.images = get_imresize_downsampled(self.image, downsampling_factor=self.downsampling_factor,
                                               downsampling_number=self.downsampling_number)
        self.images_torch = [np_to_torch(image).to(dtype=self.dtype, device=self.device)
                             for image in self.images]
        assert self.bg_hint.shape[1:] == self.image.shape[1:], (self.bg_hint.shape[1:], self.image.shape[1:])
        self.bg_hints = get_imresize_downsampled(self.bg_hint, downsampling_factor=self.downsampling_factor,
                                                 downsampling_number=self.downsampling_number)
        self.bg_hints_torch = [
            np_to_torch(bg_hint).to(dtype=self.dtype, device=self.device) for bg_hint in
            self.bg_hints]

        assert self.fg_hint.shape[1:] == self.image.shape[1:]
        self.fg_hints = get_imresize_downsampled(self.fg_hint, downsampling_factor=self.downsampling_factor,
                                                 downsampling_number=self.downsampling_number)
        self.fg_hints_torch = [
            np_to_torch(fg_hint).to(dtype=self.dtype, device=self.device) for fg_hint in
            self.fg_hints]

    def _init_losses(self):
        """
        Initializes loss functions for optimization.
        """
        dtype = self.dtype
        device = self.device
        self.l1_loss = nn.L1Loss().to(dtype=dtype, device=device)
        self.extended_l1_loss = ExtendedL1Loss().to(dtype=dtype, device=device)
        self.gradient_loss = GradientLoss().to(dtype=dtype, device=device)
        self.gray_loss = GrayLoss().to(dtype=dtype, device=device)

    def _init_nets(self):
        """
        Initializes the neural networks for dehazing:
        - left_net: Predicts the first layer.
        - mask_net: Predicts the mask (no binary).
        -right_net: Predict the second layer
        """
        pad = 'reflection'
        left_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.left_net = left_net.to(dtype=self.dtype, device=self.device)

        right_net = skip(
            self.input_depth, 3,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            upsample_mode='bilinear',
            filter_size_down=3,
            filter_size_up=3,
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.right_net = right_net.to(dtype=self.dtype, device=self.device)

        mask_net = skip_mask(
            self.input_depth, 1,
            num_channels_down=[8, 16, 32],
            num_channels_up=[8, 16, 32],
            num_channels_skip=[0, 0, 0],
            filter_size_down=3,
            filter_size_up=3,
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

        self.mask_net = mask_net.to(dtype=self.dtype, device=self.device)

    def _init_parameters(self):
        """
        Collects trainable parameters from all relevant networks.
        """
        self.parameters = [p for p in self.left_net.parameters()] + \
                          [p for p in self.right_net.parameters()] + \
                          [p for p in self.mask_net.parameters()]

    def _init_noise(self):
        """
        Initializes nets whith random noise
        """
        input_type = 'noise'
        self.left_net_inputs = [get_noise(self.input_depth,
                                          input_type,
                                          (image.shape[2], image.shape[3])).to(dtype=self.dtype,
                                                                               device=self.device).detach()
                                for image in self.images_torch]
        self.right_net_inputs = self.left_net_inputs
        input_type = 'noise'
        self.mask_net_inputs = [get_noise(self.input_depth,
                                          input_type,
                                          (image.shape[2], image.shape[3])).to(dtype=self.dtype,
                                                                               device=self.device).detach()
                                for image in self.images_torch]

    def optimize(self):
        """
        Performs the optimization process to dehaze the image.
        - Uses Adam optimizer with the specified learning rate.
        """
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # step 1
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.first_step_iter_num):
            optimizer.zero_grad()
            self._step1_optimization_closure(j)
            self._finalize_iteration()
            if self.plot_during_training:
                self._iteration_plot_closure(j, 1)
            optimizer.step()
        self._update_result_closure(1)
        if self.plot_during_training:
            self._step_plot_closure(1)

        # step 2
        optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)
        for j in range(self.second_step_iter_num):
            optimizer.zero_grad()
            self._step2_optimization_closure(j)
            self._finalize_iteration()
            if self.second_step_done:
                break
            if self.plot_during_training:
                self._iteration_plot_closure(j, 2)
            optimizer.step()
        self._update_result_closure(2)
        if self.plot_during_training:
            self._step_plot_closure(2)

    def _step1_optimization_closure(self, iteration):
        """
        The real iteration is (step-1) * self.num_iter_per_step + iteration
        """
        self._initialize_any_step(iteration)
        self._step1_optimize_with_hints()

    def _step2_optimization_closure(self, iteration):
        """
        The real iteration is (step-1) * self.num_iter_per_step + iteration
        """
        self._initialize_any_step(iteration)
        self._step2_optimize_with_hints(iteration)

    def _step1_optimize_with_hints(self):
        """
        Computes the loss and backpropagates the gradients.
        """
        self.total_loss += sum(self.extended_l1_loss(left_net_output, image_torch, fg_hint) for
                               left_net_output, fg_hint, image_torch
                               in zip(self.left_net_outputs, self.fg_hints_torch, self.images_torch))
        self.total_loss += sum(self.extended_l1_loss(right_net_output, image_torch, bg_hint) for
                               right_net_output, bg_hint, image_torch
                               in zip(self.right_net_outputs, self.bg_hints_torch, self.images_torch))

        self.total_loss += sum(self.l1_loss(((fg_hint - bg_hint) + 1) / 2, mask_net_output) for
                               fg_hint, bg_hint, mask_net_output in
                               zip(self.fg_hints_torch, self.bg_hints_torch, self.mask_net_outputs))
        self.total_loss.backward(retain_graph=True)

    def _step2_optimize_with_hints(self, iteration):
        """
        Computes the loss and backpropagates the gradients.
        """
        if iteration <= 1000:
            self.total_loss += sum(self.extended_l1_loss(left_net_output, image_torch, fg_hint) for
                                   left_net_output, fg_hint, image_torch
                                   in zip(self.left_net_outputs, self.fg_hints_torch, self.images_torch))
            self.total_loss += sum(self.extended_l1_loss(right_net_output, image_torch, bg_hint) for
                                   right_net_output, bg_hint, image_torch
                                   in zip(self.right_net_outputs, self.bg_hints_torch, self.images_torch))

        for left_out, right_out, mask_out, original_image_torch in zip(self.left_net_outputs,
                                                                       self.right_net_outputs,
                                                                       self.mask_net_outputs,
                                                                       self.images_torch):
            self.total_loss += 0.5 * self.l1_loss(mask_out * left_out + (1 - mask_out) * right_out,
                                                  original_image_torch)
            self.current_gradient = self.gray_loss(mask_out)
            iteration = min(iteration, 1000)
            self.total_loss += (0.001 * (iteration // 100)) * self.current_gradient
        self.total_loss.backward(retain_graph=True)

    def _initialize_any_step(self, iteration):
        if iteration == self.second_step_iter_num - 1:
            reg_noise_std = 0
        elif iteration < 1000:
            reg_noise_std = (1 / 1000.) * (iteration // 100)
        else:
            reg_noise_std = 1 / 1000.
        right_net_inputs = []
        left_net_inputs = []
        mask_net_inputs = []
        # creates left_net_inputs and right_net_inputs by adding small noise
        for left_net_original_input, right_net_original_input, mask_net_original_input \
                in zip(self.left_net_inputs, self.right_net_inputs, self.mask_net_inputs):
            left_net_inputs.append(
                left_net_original_input + (left_net_original_input.clone().normal_() * reg_noise_std))
            right_net_inputs.append(
                right_net_original_input + (right_net_original_input.clone().normal_() * reg_noise_std))
            mask_net_inputs.append(
                mask_net_original_input + (mask_net_original_input.clone().normal_() * reg_noise_std))
        # applies the nets
        self.left_net_outputs = [self.left_net(left_net_input) for left_net_input in left_net_inputs]
        self.right_net_outputs = [self.right_net(right_net_input) for right_net_input in right_net_inputs]
        self.mask_net_outputs = [self.mask_net(mask_net_input) for mask_net_input in mask_net_inputs]
        self.total_loss = 0

    def _finalize_iteration(self):
        left_out_np = torch_to_np(self.left_net_outputs[0])
        right_out_np = torch_to_np(self.right_net_outputs[0])
        original_image = self.images[0]
        mask_out_np = torch_to_np(self.mask_net_outputs[0])
        self.current_ssim = structural_similarity(original_image.astype(mask_out_np.dtype),
                                     mask_out_np * left_out_np + (1 - mask_out_np) * right_out_np, channel_axis=0,
                                     data_range=1.0)

        self.ssims.append(self.current_ssim)

    def _iteration_plot_closure(self, iter_number, step_number):
        """
        Displays training progress by printing loss and saving intermediate images.
        Args:
            iter_number (int): iteration number
            step_number (int): step number
        """

        if self.current_gradient is not None:
            sys.stdout.write(f'\r Step {step_number} Iteration {iter_number:5d} '
                             f'total_loss {self.total_loss.item():.5f} '
                             f'grad {self.current_gradient.item():.5f} '
                             f'SSIM {self.current_ssim:.5f}')
        else:
            sys.stdout.write(f'\rStep {step_number} Iteration {iter_number:5d} '
                             f'total_loss {self.total_loss.item():.5f} '
                             f'SSIM {self.current_ssim:.5f}')
        sys.stdout.flush()

        if iter_number % self.show_every == self.show_every - 1:
            self._plot_with_name(iter_number, step_number)

    def _update_result_closure(self,step):
        self._finalize_iteration()
        self._fix_mask()
        self.current_result = SegmentationResult(mask=self.fixed_masks[0],
                                                 left=torch_to_np(self.left_net_outputs[0]),
                                                 right=torch_to_np(self.right_net_outputs[0]),
                                                 learned_mask=torch_to_np(self.mask_net_outputs[0]),
                                                 ssim=self.current_ssim)
        if self.best_result is None or self.best_result.ssim <= self.current_result.ssim:
            self.best_result = self.current_result
        if step == 1:
            num_iters = self.first_step_iter_num
        else:
            num_iters = self.second_step_iter_num

        save_graph(self.image_name + "_step_{}_ssim".format(step), self.ssims, num_iters)
        self.ssims = []

    def _fix_mask(self):
        """
        fixing the masks using soft matting
        :return:
        """
        masks_np = [torch_to_np(mask) for mask in self.mask_net_outputs]
        new_mask_nps = [np.array([guidedFilter(image_np.transpose(1, 2, 0).astype(np.float32),
                                               mask_np[0].astype(np.float32), 50, 1e-4)])
                        for image_np, mask_np in zip(self.images, masks_np)]

        def to_bin(x):
            v = np.zeros_like(x)
            v[x > 0.5] = 1
            return v

        self.fixed_masks = [to_bin(m) for m in new_mask_nps]

    def _plot_with_name(self, iteration, step):
        for left_out, right_out, mask_out, image in zip(self.left_net_outputs,
                                                        self.right_net_outputs,
                                                        self.mask_net_outputs, self.images):
            plot_image_grid("left_right_{}_step_{}".format(iteration, step),
                            [np.clip(torch_to_np(left_out), 0, 1),
                             np.clip(torch_to_np(right_out), 0, 1)])
            mask_out_np = torch_to_np(mask_out)
            plot_image_grid("learned_mask_{}_step_{}".format(iteration, step),
                            [np.clip(mask_out_np, 0, 1), 1 - np.clip(mask_out_np, 0, 1)])

            plot_image_grid("learned_image_{}_step_{}".format(iteration, step),
                            [np.clip(mask_out_np * torch_to_np(left_out) + (1 - mask_out_np) * torch_to_np(right_out),
                                     0, 1), image])

    def _step_plot_closure(self, step_number):
        self._plot_with_name("final", step_number)

    def finalize(self):
        """
        Finalizes the segmentation process and saves the results.
        """
        save_image(self.image_name + "_left", self.best_result.left)
        save_image(self.image_name + "_learned_mask", self.best_result.learned_mask)
        save_image(self.image_name + "_right", self.best_result.right)
        save_image(self.image_name + "_original", self.images[0])
        save_image(self.image_name + "_mask", self.best_result.mask)
        learned_image = self.best_result.left * self.best_result.learned_mask + (
                1 - self.best_result.learned_mask) * self.best_result.right
        save_image(self.image_name + "_learned_image", learned_image)
        save_image("fg_hint", self.fg_hint)
        save_image("bg_hint", self.bg_hint)


def main_segmentation(image_path, conf_params={}):
    i = prepare_image(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    s = Segmentation(image_name, i, **conf_params)
    s.optimize()
    s.finalize()


if __name__ == "__main__":
    set_gpu_or_cpu_and_dtype(use_gpu=True, torch_dtype=torch.float32)

    conf_params = {
        "show_every": 500,
        "first_step_iter_num": 1000,
        "second_step_iter_num": 1000,
    }

    main_segmentation('images/mountain.jpg', conf_params=conf_params)
