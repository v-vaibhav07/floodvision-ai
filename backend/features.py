import cv2
import numpy as np

def local_mean_std(channel, window_size):
    mean = cv2.boxFilter(
        channel, cv2.CV_32F,
        (window_size, window_size)
    )

    mean_square = cv2.boxFilter(
        channel * channel, cv2.CV_32F,
        (window_size, window_size)
    )

    std = np.sqrt(np.maximum(mean_square - mean * mean, 0))

    return mean, std


GABOR_KERNELS = []

for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
    kernel = cv2.getGaborKernel(
        (15, 15),
        3.0,
        theta,
        8.0,
        0.5,
        0,
        ktype=cv2.CV_32F
    )

    kernel = kernel - kernel.mean()
    kernel = kernel / (np.sum(np.abs(kernel)) + 1e-6)

    GABOR_KERNELS.append(kernel)


def extract_teacher_features(image):

    h, w = image.shape[:2]

    rgb = image.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv = hsv / np.array([179.0, 255.0, 255.0], dtype=np.float32)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0

    channels = []

    for i in range(3):
        channels.append(rgb[:, :, i])

    for i in range(3):
        channels.append(hsv[:, :, i])

    for i in range(3):
        channels.append(lab[:, :, i])

    for sigma in [1, 2, 4]:
        blurred = cv2.GaussianBlur(
            rgb,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma
        )

        for i in range(3):
            channels.append(blurred[:, :, i])

    luminance = lab[:, :, 0]

    for window in [7, 15]:
        mean, std = local_mean_std(luminance, window)
        channels.append(mean)
        channels.append(std)

    grad_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)

    sobel_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    laplacian = np.abs(
        cv2.Laplacian(luminance, cv2.CV_32F, ksize=3)
    )

    channels.append(sobel_magnitude)
    channels.append(laplacian)

    for kernel in GABOR_KERNELS:
        response = cv2.filter2D(
            luminance,
            cv2.CV_32F,
            kernel
        )

        channels.append(np.abs(response))

    row_position = np.broadcast_to(
        np.linspace(0, 1, h, dtype=np.float32)[:, None],
        (h, w)
    )

    col_position = np.broadcast_to(
        np.linspace(0, 1, w, dtype=np.float32)[None, :],
        (h, w)
    )

    channels.append(row_position)
    channels.append(col_position)

    features = np.stack(channels, axis=-1)

    return features.reshape(-1, features.shape[-1]).astype(np.float32)


def extract_student_features(image):

    h, w = image.shape[:2]

    rgb = image.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

    saturation = hsv[:, :, 1] / 255.0

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0

    luminance = lab[:, :, 0]

    _, local_std = local_mean_std(luminance, 7)

    row_position = np.broadcast_to(
        np.linspace(0, 1, h, dtype=np.float32)[:, None],
        (h, w)
    )

    channels = [
        rgb[:, :, 0],
        rgb[:, :, 1],
        rgb[:, :, 2],
        saturation,
        local_std,
        row_position
    ]

    features = np.stack(channels, axis=-1)

    return features.reshape(-1, features.shape[-1]).astype(np.float32)


