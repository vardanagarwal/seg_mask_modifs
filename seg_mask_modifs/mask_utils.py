import cv2


def invert_mask(mask):
    """Function to invert the mask

    Args:
        mask (np.uint8): Input mask

    Returns:
        mask (np.uint8): Inverted mask
    """
    return cv2.bitwise_not(mask)


def combine_masks(mask, mask1, *masks):
    """Combine multiple masks

    Args:
        mask (np.uint8): Input mask
        mask1 (np.uint8): Input mask
        *masks (np.uint8): Any number of masks

    Returns:
        combined_mask (np.uint8): Combined mask
    """
    combined_mask = cv2.bitwise_or(mask, mask1)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask


def intersection_masks(mask, mask1, *masks):
    """Intersection of multiple masks

    Args:
        mask (np.uint8): Input mask
        mask1 (np.uint8): Input mask
        *masks (np.uint8): Any number of masks

    Returns:
        intersection_mask (np.uint8): Intersection mask
    """
    intersection_mask = cv2.bitwise_and(mask, mask1)
    for mask in masks:
        intersection_mask = cv2.bitwise_and(intersection_mask, mask)
    return intersection_mask


def dilate_mask(mask, kernel_size):
    """Dilate size of mask

    Args:
        mask (np.uint8): Input mask
        kernel_size (int or tuple): size of kernel to dilate mask

    Raises:
        ValueError: Kernel size should be an integer or tuple.

    Returns:
        dilated_mask (np.uint8): Dilated mask
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif not isinstance(kernel_size, tuple):
        raise ValueError('Kernel size should be an integer or a tuple')
    dilated_mask = cv2.dilate(mask, kernel_size)
    return dilated_mask


def erode_mask(mask, kernel_size):
    """Erode size of mask

    Args:
        mask (np.uint8): Input mask
        kernel_size (int or tuple): size of kernel to erode mask

    Raises:
        ValueError: Kernel size should be an integer or tuple.

    Returns:
        erode_mask (np.uint8): Eroded mask
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif not isinstance(kernel_size, tuple):
        raise ValueError('Kernel size should be an integer or a tuple')
    erode_mask = cv2.erode(mask, kernel_size)
    return erode_mask
