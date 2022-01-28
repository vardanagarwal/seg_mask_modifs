import json

data = json.load(open('model_utils/labels.json'))


def maskrcnn_coco():
    """ Print maskrcnn labels

    Returns: list of labels"""

    labels = data['maskrcnn_coco_labels'][1:]
    new_labels = []
    for label in labels:
        if label != "N/A":
            new_labels.append(label)
    print(', '.join(new_labels))

    return new_labels


def deeplab_pascal():
    """ Print deeplab labels

    Returns: list of labels"""

    labels = data['deeplab_pascal_labels'][1:]
    print(', '.join(labels))

    return labels


def face():
    """ Print face labels

    Returns: list of labels"""

    labels = data['face_labels'][1:]
    print(', '.join(labels))

    return labels


def all():
    """ Print labels of all models"""

    print('MaskRCNN labels')
    maskrcnn_coco()

    print('Deeplabv3 labels')
    deeplab_pascal()

    print('Face model labels')
    face()


if __name__ == '__main__':
    maskrcnn_coco()
    deeplab_pascal()
    face()
    all()
