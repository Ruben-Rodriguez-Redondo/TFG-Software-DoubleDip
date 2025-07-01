import sys
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from cv2.ximgproc import guidedFilter
from scipy.spatial.distance import cdist
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity
from tqdm import tqdm

from double_dip_core.net import *
from double_dip_core.net.downsampler import *
from double_dip_core.net.losses import ExtendedL1Loss, GrayLoss
from double_dip_core.net.noise import get_noise
from double_dip_core.utils.image_io import *


def equation_3(img, r, c, rows, cols, K=64, C=3):
    """
    Computes Equation (3) from the paper: Context Aware Saliency Detection
    Saliency at a pixel is defined based on the average color+spatial distance
    to the K most similar pixels in the image.

    Args:
        img (ndarray): Input image in range [0, 1], shape (rows, cols, 3).
        r (int): Row index of the target pixel.
        c (int): Column index of the target pixel.
        rows (int): Total number of image rows.
        cols (int): Total number of image columns.
        K (int): Number of closest pixels to consider (default: 64).
        C (float): Weight for spatial distance (default: 3).

    Returns:
        float: Saliency score in [0, 1] for the pixel at position (r, c).
    """

    c0 = img[r, c]  # Color vector of the reference pixel

    # Normalized relative spatial positions
    row_coords, col_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    d_rows = (r - row_coords) / rows
    d_cols = (c - col_coords) / cols
    spatial_dists = np.sqrt(d_rows ** 2 + d_cols ** 2)  # Euclidean spatial distance

    # Color distance from the reference pixel to every other pixel
    color_diffs = np.linalg.norm(img - c0, axis=2)  # Euclidean color distance

    # Final combined distance as defined in Equation (1)
    total_dists = color_diffs / (1 + C * spatial_dists)

    # Flatten and select the K smallest distances
    flat = total_dists.flatten()
    flat.sort()
    topk = flat[:min(K, flat.shape[0])]

    # Final saliency value based on Equation (3)
    return 1 - np.exp(-np.mean(topk))


def add_blur_and_call_eq3(src, u, rows, cols):
    """
    Applies optional blurring to the input image and computes the saliency map
    by evaluating Equation (3) at every pixel position.

    Equation (3) defines pixel saliency as the average color+spatial dissimilarity
    to the K nearest patches, and is computed using `equation_3`.

    Args:
        src (ndarray): Input image in range [0, 1], shape (rows, cols, 3).
        u (int): Blur radius (0 = no blur). Applied using uniform averaging.
        rows (int): Number of rows in the image.
        cols (int): Number of columns in the image.

    Returns:
        ndarray: Normalized saliency map in range [0, 1], shape (rows, cols).
    """
    # Apply blur with window size (2u + 1)
    if u > 0:
        blurred = cv2.blur(src, (2 * u + 1, 2 * u + 1))
    else:
        blurred = src.copy()

    # Compute saliency for each row in parallel
    def compute_row(row):
        return [equation_3(blurred, row, col, rows, cols) for col in range(cols)]

    results = [None] * rows
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(compute_row, r): r for r in range(rows)}
        for future in tqdm(as_completed(futures), file=sys.stdout, total=rows, desc=f"Saliency map u={u}"):
            r = futures[future]
            results[r] = future.result()

    # Stack results and normalize to [0, 1]
    saliency = np.array(results, dtype=np.float32)
    saliency /= saliency.max()
    return saliency


def saliency_hints(img, u=4):
    """
    Computes saliency-based segmentation hints using the formulation of Equations (4) and (5)
    from the Context-Aware Saliency Detection approach.

    The process includes:
    - Conversion to Lab color space and downscaling
    - Saliency computation at multiple blur levels (Equation 4)
    - Refinement by proximity to highly salient regions (Equation 5)

    Args:
        img (ndarray): Input image in shape (C, H, W), with values in [0, 255].
        u (int): Blur radius to define the size of the local averaging window.

    Returns:
        ndarray: Refined saliency map in shape (H, W), dtype uint8 in [0, 255].
    """
    # Reorder to (H, W, C) and resize image for efficiency
    src = np.transpose(img, (1, 2, 0))  # Convert to (H, W, C)
    h, w = src.shape[:2]
    scale = min(200 / max(h, w), 1.0)  # 200
    new_w = int(w * scale)
    new_h = int(h * scale)
    ori_shape = (w, h)
    src = cv2.resize(src, (new_w, new_h))

    # Convert to Lab color space in range [0, 1]
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2Lab).astype(np.float32) / 255.0
    rows, cols = lab.shape[:2]

    # Compute saliency maps at multiple blur scales (Equation 4)
    tg0 = add_blur_and_call_eq3(lab, 0, rows, cols)
    tg2 = add_blur_and_call_eq3(lab, u // 2, rows, cols)
    tg4 = add_blur_and_call_eq3(lab, u, rows, cols)
    tg_avg = (tg0 + tg2 + tg4) / 3.0

    # Identify high-saliency pixels (top 10%)
    threshold = np.percentile(tg_avg, 90)
    main_part = np.argwhere(tg_avg > threshold)
    S = tg_avg.copy()

    if len(main_part) > 0:
        # Compute spatial coordinates of all pixels
        row_coords, col_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        all_coords = np.stack((row_coords.ravel(), col_coords.ravel()), axis=-1)  # (rows*cols, 2)

        # Normalize coordinates
        all_coords_norm = all_coords / np.array([rows, cols])
        main_part_norm = main_part / np.array([rows, cols])

        # Compute distance to nearest salient region (Equation 5)
        dists = cdist(all_coords_norm, main_part_norm)  # (rows*cols, N)
        min_dists = np.min(dists, axis=1)

        # Attenuate saliency for low-saliency areas based on distance
        S_flat = S.ravel()
        mask = S_flat <= threshold
        S_flat[mask] *= (1.0 - min_dists[mask])
        S = S_flat.reshape((rows, cols))

    # Resize back to original shape and convert to uint8
    return cv2.resize((S * 255).astype(np.uint8), ori_shape)


def obtain_bg_fg_segmentation_hints(img):
    sal = saliency_hints(img)

    # Foreground mhint
    image_eq = cv2.equalizeHist(sal)
    fg = np.zeros_like(image_eq)
    fg[image_eq > (255 - 15.5)] = 255

    # 2. Background hint
    image_eq2 = cv2.equalizeHist(sal)
    bg = np.zeros_like(image_eq2)
    bg[image_eq2 <= 15.5] = 255

    eq_sal = pil_to_np(cv2.equalizeHist(sal))

    return fg, bg, eq_sal


SegmentationResult = namedtuple("SegmentationResult",
                                ['mask', 'learned_mask', 'left', 'right', 'ssim', 'psnr'])


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
            fg, bg, eq_sal = obtain_bg_fg_segmentation_hints(self.image.copy())
            self.fg_hint = pil_to_np(fg)
            self.bg_hint = pil_to_np(bg)
            self.eq_sal = eq_sal

        self.image_name = image_name
        self.plot_during_training = plot_during_training
        self.downsampling_factor = downsampling_factor
        self.downsampling_number = downsampling_number
        self.ssims = []
        self.psnrs = []
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
        self.gray_loss = GrayLoss().to(dtype=dtype, device=device)

    def _init_nets(self):
        """
        Initializes the neural networks for segmentation:
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

        mask_net = skip(
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
        self._initialize_any_step(iteration)
        self._step1_optimize_with_hints()

    def _step2_optimization_closure(self, iteration):

        self._initialize_any_step(iteration)
        self._step2_optimize_with_hints(iteration)

    def _step1_optimize_with_hints(self):
        """
        Computes the loss and backpropagates the gradients.
        """
        # Reconstruction loss
        self.total_loss += sum(self.extended_l1_loss(left_net_output, image_torch, fg_hint) for
                               left_net_output, fg_hint, image_torch
                               in zip(self.left_net_outputs, self.fg_hints_torch, self.images_torch))
        self.total_loss += sum(self.extended_l1_loss(right_net_output, image_torch, bg_hint) for
                               right_net_output, bg_hint, image_torch
                               in zip(self.right_net_outputs, self.bg_hints_torch, self.images_torch))
        # Mask regularization loss
        self.total_loss += sum(self.l1_loss(((fg_hint - bg_hint) + 1) / 2, mask_net_output) for
                               fg_hint, bg_hint, mask_net_output in
                               zip(self.fg_hints_torch, self.bg_hints_torch, self.mask_net_outputs))
        self.total_loss.backward(retain_graph=True)

    def _step2_optimize_with_hints(self, iteration):
        """
        Computes the loss and backpropagates the gradients.
        """
        if iteration <= 1000:
            # Reconstruction Loss
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
            # Reconstruiction loss
            self.total_loss += 0.5 * self.l1_loss(mask_out * left_out + (1 - mask_out) * right_out,
                                                  original_image_torch)
            # Regularization loss (no exclussion loss)
            self.current_gradient = self.gray_loss(mask_out)
            self.total_loss += (0.001 * (min(iteration, 1000) // 100)) * self.current_gradient
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

        reconstruc = mask_out_np * left_out_np + (1 - mask_out_np) * right_out_np
        self.current_ssim = structural_similarity(original_image.astype(mask_out_np.dtype),
                                                  reconstruc,
                                                  channel_axis=0,
                                                  data_range=1.0)
        self.current_psnr = compute_psnr(original_image.astype(mask_out_np.dtype),
                                         reconstruc, data_range=1.0)
        self.ssims.append(self.current_ssim)
        self.psnrs.append(self.current_psnr)

    def _iteration_plot_closure(self, iter_number, step_number):
        """
        Displays training progress by printing loss and saving intermediate images_remove.
        Args:
            iter_number (int): iteration number
            step_number (int): step number
        """

        sys.stdout.write(f'\rStep {step_number}, '
                         f'Iteration {iter_number + 1}, '
                         f'Loss: {self.total_loss.item():.5f}, '
                         f'SSIM: {self.current_ssim:.3f}, '
                         f'PSNR: {self.current_psnr:.3f}')
        sys.stdout.flush()

        if iter_number % self.show_every == self.show_every - 1:
            self._plot_with_name(iter_number, step_number)

    def _update_result_closure(self, step):
        self._finalize_iteration()
        self._fix_mask()
        self.current_result = SegmentationResult(mask=self.fixed_masks[0],
                                                 left=torch_to_np(self.left_net_outputs[0]),
                                                 right=torch_to_np(self.right_net_outputs[0]),
                                                 learned_mask=torch_to_np(self.mask_net_outputs[0]),
                                                 ssim=self.current_ssim,
                                                 psnr=self.current_psnr)
        if self.best_result is None or self.best_result.ssim <= self.current_result.ssim:
            self.best_result = self.current_result

    def _fix_mask(self):
        """
        fixing the masks using soft matting
        :return:
        """
        masks_np = [torch_to_np(mask) for mask in self.mask_net_outputs]
        new_mask_nps = [np.array([guidedFilter(image_np.transpose(1, 2, 0).astype(np.float32),
                                               mask_np[0].astype(np.float32), 5, 1e-4)])
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
        save_image("equalize_saliency", self.eq_sal)

        save_graph(self.image_name + "_ssim", self.ssims, self.first_step_iter_num + self.second_step_iter_num,
                   title="SSIM")
        save_graph(self.image_name + "_psnr", self.psnrs, self.first_step_iter_num + self.second_step_iter_num,
                   title="PSNR")


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
        "first_step_iter_num": 2000,
        "second_step_iter_num": 4000,
    }
    start = time.time()
    main_segmentation('images/segmentation/segmentation_2.png', conf_params=conf_params)
    print(f'\nTime: {time.time() - start} seconds')
