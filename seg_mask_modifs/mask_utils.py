import cv2
import numpy as np

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

def __mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function for OpenCV window. Records clicked points and draws a circle at the clicked position.

    Args:
        event (int): OpenCV event type.
        x (int): x-coordinate of the mouse click.
        y (int): y-coordinate of the mouse click.
        flags (int): Additional flags for the callback.
        param (Any): Additional parameters for the callback.
    """
    global clicked_points, image_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        color = image_copy[y, x]

        red_threshold = 30
        is_red = abs(color[0] - 0) < red_threshold and \
                 abs(color[1] - 0) < red_threshold and \
                 abs(color[2] - 255) < red_threshold

        circle_color = (0, 255, 0) if is_red else (0, 0, 255)
        cv2.circle(image_copy, (x, y), 5, circle_color, -1)
        cv2.imshow("Input Image", image_copy)

def __extract_connected_component(image, point):
    """
    Extracts the connected component from the image based on the given point.

    Args:
        image (np.uint8): Input image.
        point (tuple): A tuple (x, y) representing the point clicked in the image.

    Returns:
        numpy.ndarray: Extracted connected component.
    """
    color = image[point[1], point[0]]
    mask = cv2.inRange(image, color, color)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    label = labels[point[1], point[0]]
    component_mask = (labels == label).astype(np.uint8) * 255
    component_mask = np.stack([component_mask] * 3, axis=-1)
    
    return component_mask & image

def __extract_regions(image, clicked_points):
    """
    Extracts regions with the same color as the clicked points in the input image.

    Args:
        image (np.uint8): Input image.
        clicked_points (list): List of clicked points as tuples (x, y).

    Returns:
        output_image (np.uint8): Output image containing only the extracted regions.
    """
    output_image = np.zeros_like(image)
    for point in clicked_points:
        component = __extract_connected_component(image, point)
        output_image = cv2.bitwise_or(output_image, component)
    return output_image

def select_mask_regions(image):
    """
    Processes the input image by displaying it, collecting clicked points, and returning the output image after
    applying the extract_regions function.

    Args:
        image (np.uint8): Input image.

    Returns:
        output_image (np.uint8): Output image containing only the extracted regions.
    """
    global clicked_points, image_copy
    image_copy = image.copy()
    clicked_points = []

    cv2.namedWindow("Input Image")
    cv2.setMouseCallback("Input Image", __mouse_callback)

    while True:
        cv2.imshow("Input Image", image_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    output_image = __extract_regions(image, clicked_points)
    return output_image


def get_binary_mask(mask):
    """Convert mask to binary mask

    Args:
        mask (np.uint8): Input mask

    Returns:
        binary_mask (np.uint8): Binary mask
    """
    binary_mask = mask.copy()
    binary_mask[binary_mask > 0] = 255
    return binary_mask