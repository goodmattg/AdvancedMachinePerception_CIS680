from torchvision import transforms


def scale_to_target(image, target_size):
    """
    Scale channels-last image to target size

    Returns:
        resize_factors: (res_w, res_h)
        padding: (top, bottom, left, right)
    """
    w_i, h_i = image.shape[:2]
    w_t, h_t = target_size

    w_ratio = w_t / w_i
    h_ratio = h_t / h_i

    if w_ratio < h_ratio:
        resize_factors = (w_ratio, w_ratio)

        pad_h = h_t - (h_i * w_ratio)
        padding = (pad_h // 2, pad_h // 2, 0, 0)

    else:
        resize_factors = (h_ratio, h_ratio)

        pad_w = w_t - (w_i * h_ratio)
        padding = (0, 0, pad_w // 2, pad_w // 2)

    padding = tuple(map(int, padding))
    return resize_factors, padding


def rcnn_preprocess():

    transforms.Compose(
        [transforms.CenterCrop(10), transforms.ToTensor(),]
    )

