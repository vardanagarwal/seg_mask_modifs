import cv2
import numpy as np


def blur_bg(img, mask, kernel_size=(5, 5)):
    """ blur the background of image (black region of mask).

    Args:
        img (np.uint8): image to blur
        mask (np.uint8): mask to use as reference
        kernel_size (tuple, optional): kernel size to use for blur. Defaults to (5, 5).

    Returns:
        img_blur (np.uint8): image with blurred background.
    """
    img_bg = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    img_bg = cv2.blur(img_bg, kernel_size)
    img_fg = cv2.bitwise_and(img, img, mask=mask)
    img_blur = cv2.add(img_bg, img_fg)
    return img_blur


def blur_fg(img, mask, kernel_size=(5, 5)):
    """blur the foreground of image (white region of mask).

    Args:
        img (np.uint8): image to blur
        mask (np.uint8): mask to use as reference
        kernel_size (tuple, optional): kernel size to use for blur. Defaults to (5, 5).

    Returns:
        img_blur (np.uint8): image with blurred foreground.
    """
    img_blur = blur_bg(img, cv2.bitwise_not(mask), kernel_size)
    return img_blur


def grayscale_bg(img, mask):
    """convert the background of image to grayscale

    Args:
        img (np.uint8): image to convert to grayscale
        mask (np.uint8): mask to use as reference

    Returns:
        img_grayscale (np.uint8): image with grayscale background.
    """
    img_bg = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)
    img_bg = cv2.merge([img_bg, img_bg, img_bg])
    img_fg = cv2.bitwise_and(img, img, mask=mask)
    img_grayscale = cv2.add(img_bg, img_fg)
    return img_grayscale


def grayscale_fg(img, mask):
    """convert the foreground of image to grayscale

    Args:
        img (np.uint8): image to convert to grayscale
        mask (np.uint8): mask to use as reference

    Returns:
        img_grayscale (np.uint8): image with grayscale foreground.
    """
    return grayscale_bg(img, cv2.bitwise_not(mask))


def replace_bg(img, mask, img2):
    """Replace the background of image with another image

    Args:
        img (np.uint8): image to convert to grayscale
        mask (np.uint8): mask to use as reference

    Returns:
        img_grayscale (np.uint8): image with background foreground.
    """
    img_bg = cv2.bitwise_and(img2, img2, mask=cv2.bitwise_not(mask))
    img_fg = cv2.bitwise_and(img, img, mask=mask)
    return cv2.add(img_bg, img_fg)


def replace_fg(img, mask, img2):
    """Replace the foreground of image with another image

    Args:
        img (np.uint8): Main image with part to replace
        mask (np.uint8): mask to use as reference
        img2 (np.uint8): Image to replace with

    Returns:
        img_grayscale (np.uint8): image with background foreground.
    """
    return replace_bg(img, cv2.bitwise_not(mask), img2)


def inpaint_bg(img, mask):
    """Inpaint the background of the image

    Args:
        img (np.uint8): Image to inpaint
        mask (np.uint8): Mask to use as reference

    Returns:
        img_inpaint(np.uint8): image with inpainted background.
    """
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    img_inpaint = np.zeros_like(img)
    cv2.xphoto.inpaint(img_lab, mask, img_inpaint, cv2.xphoto.INPAINT_SHIFTMAP)
    img_inpaint = cv2.cvtColor(img_inpaint, cv2.COLOR_Lab2BGR)
    return img_inpaint


def inpaint_fg(img, mask):
    """Inpaint the foreground of the image

    Args:
        img (np.uint8): Image to inpaint
        mask (np.uint8): Mask to use as reference

    Returns:
        img_inpaint(np.uint8): image with inpainted foreground.
    """
    return inpaint_bg(img, cv2.bitwise_not(mask))


def outline(img, mask, color=(0, 0, 0), thickness=2):
    """Draw the outline of the contours on the given image

    Args:
        img (np.uint8): Image to draw outline on
        mask (np.uint8): mask to use as reference
        color (tuple, optional): Color to draw outline with (in BGR). Defaults to (0, 0, 0).
        thickness (int, optional): thickness of the outline drawn. Defaults to 2.

    Returns:
        img(np.uint8): image with outline drawn
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, color, thickness)
    return img


def transparent_bg(img, mask):
    """Convert the background of image to transparent.

    Args:
        img (np.uint8): Image to convert to transparent
        mask (np.uint8): Mask to use as reference

    Returns:
        img_transparent (np.uint8): image with transparent background
    """
    img = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = cv2.split(img)
    img_transparent = cv2.merge([b, g, r, mask], 4)
    return img_transparent


def transparent_fg(img, mask):
    """Convert the foreground of image to transparent.

    Args:
        img (np.uint8): Image to convert to transparent
        mask (np.uint8): Mask to use as reference

    Returns:
        img_transparent (np.uint8): image with transparent foreground
    """
    return transparent_bg(img, cv2.bitwise_not(mask))


def pixelate_bg(img, mask, kernel=(16, 16)):
    """Pixelate the background of image

    Args:
        img (np.uint8): Image to pixelate
        mask (np.uint8): Mask to use as reference
        kernel (tuple, optional): Pixelate size, the smaller the value the greater the effect. Defaults to (16, 16).

    Returns:
        img_pixelate (np.uint8): Image with pixelated background
    """
    h, w = img.shape[:2]
    temp = cv2.resize(img, kernel)
    img_res = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    img_pixelate_bg = cv2.bitwise_and(
        img_res, img_res, mask=cv2.bitwise_not(mask))
    img_pixelate_fg = cv2.bitwise_and(img, img, mask=mask)
    img_pixelate = cv2.add(img_pixelate_bg, img_pixelate_fg)
    return img_pixelate


def pixelate_fg(img, mask, kernel=(16, 16)):
    """Pixelate the foreground of image

    Args:
        img (np.uint8): Image to pixelate
        mask (np.uint8): Mask to use as reference
        kernel (tuple, optional): Pixelate size, the smaller the value the greater the effect. Defaults to (16, 16).

    Returns:
        img_pixelate (np.uint8): Image with pixelated foreground
    """
    return pixelate_bg(img, cv2.bitwise_not(mask), kernel)


if __name__ == "__main__":
    # import numpy as np
    from mask_generator import mask_generator
    obj = mask_generator()
    img = cv2.imread('images/face.jpg')
    img = cv2.resize(img, None, fx=2, fy=2)
    mask = obj.generate(img, ["face"])
    h, w = img.shape[:2]
    # mask = np.zeros((h, w), dtype=np.uint8)
    # mask[int(h/3): int(2*h/3), int(w/3): int(2*w/3)] = 255
    # cv2.imshow('original', img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('img', grayscale_bg(img, mask))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('images/city_grayscale_bg.jpg', grayscale_bg(img, mask))
    # cv2.imwrite('images/city_outline.jpg', outline(img, mask))
    # cv2.imwrite('images/city_pixelate_fg.jpg', pixelate_fg(img, mask))
    cv2.imwrite('images/face_blur_fg.jpg', blur_fg(img, mask, (15, 15)))
