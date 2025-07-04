import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from skimage.metrics import structural_similarity as ssim

matplotlib.use('TkAgg')


def crop_image(img, d=32):
    """
    Crops a NumPy image to ensure its dimensions are divisible by `d`.    Args:
        pil img: image in pil format
        d: numer which image shape going to be multiples of
    Returns:
        img_cropped: the cropped image
    """

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def crop_frame(frame, d=32):
    """
    Crops a NumPy frame (from a video) to ensure its dimensions are divisible by `d`.

    Args:
        frame: NumPy array (H, W, C) representing the image.
        d: int, number that image dimensions should be a multiple of.

    Returns:
        frame_cropped: Cropped NumPy image.
    """
    h, w, c = frame.shape
    new_h = (h // d) * d
    new_w = (w // d) * d

    # Ajustar los recortes para centrar la imagen
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2

    frame_cropped = frame[y1:y1 + new_h, x1:x1 + new_w, :]
    return frame_cropped


def get_image_grid(images_np, nrow=8):
    """
    Creates a grid from a list of images_remove by concatenating them.
    Args:
        images_np: List of images_remove as NumPy arrays.
        nrow: Number of images_remove per row in the grid.
    Returns:
        NumPy array representing the image grid.
    """
    images_torch = [torch.from_numpy(x).type(torch.FloatTensor) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)

    return torch_grid.numpy()


def plot_image_grid(name, images_np, interpolation='lanczos', output_path="output/"):
    """
    Plots and saves a grid of two images_remove.
    Args:
        name: Filename for saving the grid.
        images_np: List of image NumPy arrays (each should be 3xHxW or 1xHxW).
        interpolation: Interpolation method for displaying images_remove.
        output_path: Directory to save the output image.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.clf()
    assert len(images_np) == 2
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images_remove should have 1 or 3 channels"

    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, 2)

    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    plt.savefig(output_path + "{}.png".format(name))


def save_image(name, image_np, output_path="output/"):
    """
    Saves a numpy array as an image file in the specified output path.
    Args:
        name: str, name of the image file.
        image_np: np.ndarray, image data in numpy format.
        output_path: str, directory where the image will be saved.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    p = np_to_pil(image_np)
    p.save(output_path + "{}.png".format(name))


def save_graph(name, graph_lists, num_iters, title=None, output_path="output/"):
    """
    Saves a line graph from a list of values.
    Args:
        name: str, output graph name.
        graph_list: list, values to plot.
        num_iters: total of iterations
        output_path: str, directory where the graph will be saved.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.clf()
    if isinstance(graph_lists[0], np.floating):
        graph_lists = [graph_lists]

    for graph_list in graph_lists:
        # Create a proportional scale for iterations
        iter_steps = np.linspace(0, num_iters, len(graph_list))
        plt.plot(iter_steps, graph_list, linestyle='-')

    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel(title, fontsize=16)
    plt.title(f"{title} vs Iteration", fontsize=18)
    # plt.legend([f"{title} {i + 1}" for i in range(len(graph_lists))])
    plt.legend([title], fontsize=14)
    plt.grid(False)

    plt.savefig(output_path + name + ".png")


def create_augmentations(np_image):
    """
    Creates different augmented versions of an image by rotating and flipping.
    Args:
        np_image: np.ndarray, input image.
    Returns:
        list: Augmented images_remove.
    """
    aug = [np_image.copy(), np.rot90(np_image, 1, (1, 2)).copy(),
           np.rot90(np_image, 2, (1, 2)).copy(), np.rot90(np_image, 3, (1, 2)).copy()]
    flipped = np_image[:, ::-1, :].copy()
    aug += [flipped.copy(), np.rot90(flipped, 1, (1, 2)).copy(), np.rot90(flipped, 2, (1, 2)).copy(),
            np.rot90(flipped, 3, (1, 2)).copy()]
    return aug


def load(path):
    """
    Loads an image from a given path.
    Args:
        path: str, path to the image file.
    Returns:
        PIL.Image: Loaded image.
    """
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """
    Loads an image and resizes it if necessary.
    Args:
        path: str, path to image.
        imsize: tuple or int, target size (-1 for no resize).
    Returns:
        tuple: (PIL image, numpy image)
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def prepare_image(file_name):
    """
    Loads and preprocesses an image to make it divisible by 32.
    Args:
        file_name: str, path to image.
    Returns:
        np.ndarray: Preprocessed image.
    """
    img_pil = crop_image(get_image(file_name, -1)[0], d=32)
    return pil_to_np(img_pil)


def save_video(video_name, video_frames, fps, output_path="output/"):
    """
    Converts a sequence of images_remove into a video.
    Args:
        images_dir: str, directory containing the images_remove.
        name: str, output video name.
        gray: bool, whether to convert images_remove to grayscale.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    _, height, width = video_frames[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    output_path = output_path + video_name + "_dehazed.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in video_frames:
        frame = np.array(np_to_pil(frame))
        out.write(frame)
    out.release()
    return output_path


def prepare_video(video_path):
    """
    Loads a video file and prepares it
    for processing dividing it into its frames.
    Args:
        video_path: Path of the video file.
    Returns:
        A cropped and normalized video tensor.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(crop_frame(frame))
    cap.release()
    frames = np.array(frames)
    frames = frames.transpose(0, 3, 1, 2)
    frames = frames.astype(np.float32) / 255.0
    return frames, fps


def video_to_frames(video_path):
    """Loads a video and returns a list of frames in BGR uint8 format."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def compute_tssim(frames):
    """
    Returns a list of SSIM values between consecutive frame pairs (tSSIM).
    Values closer to 1 indicate higher temporal coherence.
    """
    tssim_vals = []
    for f1, f2 in zip(frames[:-1], frames[1:]):
        # Normalize to 0–1 and use channel_axis (skimage >= 0.19)
        ssim_val = ssim(f1 / 255.0, f2 / 255.0, channel_axis=2, data_range=1.0)
        tssim_vals.append(ssim_val)
    return np.array(tssim_vals)


def analyze_tssim_videos(video_paths, output_path="output/"):
    """
    Computes and plots tSSIM values, mean, and frame-to-frame fluctuation
    for a list of videos. Saves the figure to disk.
    """
    titles = ["Haze video", "Dehaze video"]
    assert len(video_paths) == len(titles), "Number of titles must match number of videos."

    tssim_list = []
    stats_list = []

    for j, path in enumerate(video_paths):
        frames = video_to_frames(path)
        tssim_vals = compute_tssim(frames)
        fluct = np.abs(np.diff(tssim_vals)).mean()
        mean_val = tssim_vals.mean()

        print(f"\nVideo: {titles[j]}")
        print(f"tSSIM mean:     {mean_val:.4f}")
        print(f"Mean fluctuation (|ΔtSSIM|): {fluct:.6f}")
        print("-" * 40)

        tssim_list.append(tssim_vals)
        stats_list.append((mean_val, fluct))

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Plotting
    fig, axs = plt.subplots(len(video_paths), 1, figsize=(10, 3 * len(video_paths)), sharex=False)
    if len(video_paths) == 1:
        axs = [axs]  # Handle case with only one subplot

    for i, (tssim_vals, stats) in enumerate(zip(tssim_list, stats_list)):
        mean_val, fluct = stats
        axs[i].plot(tssim_vals, color='tab:blue')
        axs[i].set_ylim(0, 1)
        axs[i].set_ylabel("tSSIM")
        axs[i].set_title(titles[i])
        axs[i].grid(False)
        axs[i].set_xlabel("Frame")

        # Text box with stats
        textstr = f"Mean tSSIM: {mean_val:.4f}\n Mean |ΔtSSIM|: {fluct:.4f}"
        axs[i].text(0.98, 0.05, textstr, transform=axs[i].transAxes,
                    fontsize=10, verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_path, "tSSIM_comparison.png")
    plt.savefig(output_file, dpi=300)
    plt.close()


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.
    From H x W x C [0...255] to C x H x W [0..1]
    Args:
        img_PIL: image Pil in format H x W x C [0...255]
    Return:
        np.array in format C x H x W [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.
    From C x W x H [0..1] to  W x H x C [0...255]
    Args:
        img_np: np.array C x W x H [0..1]
    Return:
        converted image
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    Args:
        img_np: Format C x W x H [0..1]
    Return:
        converted image
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    Args:
        img_var: Format 1 x C x W x H [0..1]
    Return:
        converted image
    """
    return img_var.detach().cpu().numpy()[0]
