from PIL import Image
import torch.utils.data as data

from data_augumentation import Compose, Scale, RandomRotation, RandomMirror, Resize, NormalizeTensor


def make_data_path_list(text_path, data_path):
    """
        Parameters
        ----------
        text_path: str
            path for text data file
        data_path: str
            path for image data file
        Returns
        -------
        train_img, train_anno, val_img, val_anno: tuple
            A list containing the path to the data
        """

    with open(text_path + 'train.txt', 'r') as f:
        train_files = f.read().split("\n")

    with open(text_path + 'val.txt', 'r') as f:
        val_files = f.read().split("\n")

    train_img = []
    train_anno = []

    for file in train_files:
        train_img.append(data_path + 'images/' + file + '.png')
        train_anno.append(data_path + 'masks/' + file + '.png')

    val_img = []
    val_anno = []

    for file in val_files:
        val_img.append(data_path + 'images/' + file + '.png')
        val_anno.append(data_path + 'masks/' + file + '.png')

    return train_img, train_anno, val_img, val_anno


class DataTransform:
    """
    Attributes
    ----------
    input_size: int
        Size of the image after resizing
    color_mean: (R, G, B)
        Average value each color channel
    color_std: (R, G, B)
        Standard deviation each color channel
    """

    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train': Compose([
                Scale(scale=[0.5, 1.5]),
                RandomRotation(angle=[-10, 10]),
                RandomMirror(),
                Resize(input_size),
                NormalizeTensor(color_mean, color_std)
            ]),
            'val': Compose([
                Resize(input_size),
                NormalizeTensor(color_mean, color_std)
            ])
        }

    def __call__(self, phase, img, anno_class_img):
        """

        Parameters
        ----------
        phase: 'train' or 'val'
        img: input image
        anno_class_img: annotation image

        Returns
        -------
        transformed image: torch.tensor(
        """
        return self.data_transform[phase](img, anno_class_img)


class VOCDataset(data.Dataset):
    """
    Class to create a VOC Dataset

    Attributes
    ----------
    img_list: list()
        A list containing the paths to the images
    anno_list: list()
        A list containing the paths to the annotation
    phase: 'train' or 'test'
        Set train or test
    transform: object
        preprocessing class instance
    """

    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        """return number of image"""
        return len(self.img_list)

    def __getitem__(self, index):
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        # load images
        img_file_path = self.img_list[index]
        img = Image.open(img_file_path)
        img = img.convert("RGB")

        # load annotation images
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)
        anno_class_img = anno_class_img.convert('L')

        # preform pretreatment
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img
