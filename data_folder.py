
import torch.utils.data as data
from torch.utils.data import Dataset
import os
from pathlib import Path
from PIL import Image
import numpy as np
import albumentations as A
import torch
from skimage import transform
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def img_loader(path, num_channels):
    if num_channels == 1:
        img = Image.open(path)
    else:
        img = Image.open(path).convert('RGB')

    return img

def my_aug(imgs_list):
    imgs = [np.array(img) for img in imgs_list]
    hybrid = A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        # A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
        ], p=0.5),
        A.GaussianBlur(p=0.5),
        A.MedianBlur(p=0.5),
        # # A.CLAHE(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        A.ColorJitter(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5)]
    )

    hybrided = hybrid(image=imgs[0], masks=imgs[1:])

    img_hybrid = hybrided['image']
    masks_hybrid = hybrided['masks']

    img =  Image.fromarray(img_hybrid)
    weight = Image.fromarray(masks_hybrid[0])
    mask = Image.fromarray(masks_hybrid[1])



    return [img, weight, mask]

# get the image list pairs
def get_imgs_list(dir_list, post_fix=None):
    """
    :param dir_list: [img1_dir, img2_dir, ...]
    :param post_fix: e.g. ['label.png', 'weight.png',...]
    :return: e.g. [(img1.ext, img1_label.png, img1_weight.png), ...]
    """
    img_list = []
    if len(dir_list) == 0:
        return img_list
    if len(dir_list) != len(post_fix) + 1:
        raise (RuntimeError('Should specify the postfix of each img type except the first input.'))
    img_filename_list = [os.listdir(dir_list[i]) for i in range(len(dir_list))]

    
    for img in img_filename_list[0]:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(img)[0]
        item = [os.path.join(dir_list[0], img),]
        for i in range(1, len(img_filename_list)):
            # img_name = '{:s}_{:s}'.format(img1_name, post_fix[i-1]) 初始的这种，记得还原
            img_name = img1_name + post_fix[i - 1]
            if img_name in img_filename_list[i]:
                img_path = os.path.join(dir_list[i], img_name)
                item.append(img_path)

        if len(item) == len(dir_list):
            img_list.append(tuple(item))

    return img_list


class DataFolder_USD(data.Dataset):
    def __init__(self, dir_list, post_fix, num_channels, data_transform=None, loader=img_loader, Training=False):
        super(DataFolder_USD, self).__init__()
        if len(dir_list) != len(num_channels):
            raise RuntimeError('Length of dir_list must equal length of num_channels.')

        print(">> dir_list:", dir_list)
        print(">> post_fix:", post_fix)

        image_dir = dir_list[0]
        label_dir = dir_list[1]

        # 获取文件名（只保留 .png / .jpg 等）
        image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(tuple(post_fix))])
        label_files = sorted([f for f in os.listdir(label_dir) if f.lower().endswith(tuple(post_fix))])

        # 取交集匹配（文件名相同部分）
        image_basenames = {os.path.splitext(f)[0] for f in image_files}
        label_basenames = {os.path.splitext(f)[0] for f in label_files}
        common_basenames = sorted(list(image_basenames.intersection(label_basenames)))

        if len(common_basenames) == 0:
            print("⚠️ 未找到匹配的图像-标签对，请检查文件名！")
            print("图像文件示例:", image_files[:5])
            print("标签文件示例:", label_files[:5])
            raise RuntimeError('Found 0 image pairs in given directories.')

        # 生成路径配对列表
        self.img_list = []
        for name in common_basenames:
            img_path = os.path.join(image_dir, name + '.png')
            label_path = os.path.join(label_dir, name + '.png')
            self.img_list.append([img_path, label_path])

        print(f"✅ 共找到 {len(self.img_list)} 对图像-标签。")

        self.data_transform = data_transform
        self.num_channels = num_channels
        self.loader = loader
        self.train = Training

    def __getitem__(self, index):
        img_paths = self.img_list[index]

        sample = [self.loader(img_paths[i], self.num_channels[i]) for i in range(len(img_paths))]

        if self.train:
            sample = my_aug(sample)

        if self.data_transform is not None:
            sample = self.data_transform(sample)

        return sample

    def __len__(self):
        return len(self.img_list)


# dataset that supports one input image, one target image, and one weight map (optional)
class DataFolder(data.Dataset):
    def __init__(self, dir_list, post_fix, num_channels, data_transform=None, loader=img_loader, Training=False):
        super(DataFolder, self).__init__()
        if len(dir_list) != len(post_fix) + 1:
            raise (RuntimeError('Length of dir_list is different from length of post_fix + 1.'))
        if len(dir_list) != len(num_channels):
            raise (RuntimeError('Length of dir_list is different from length of num_channels.'))

        print(">> dir_list:", dir_list)
        print(">> post_fix:", post_fix)
        self.img_list = get_imgs_list(dir_list, post_fix)
        if len(self.img_list) == 0:
            raise(RuntimeError('Found 0 image pairs in given directories.'))

        self.data_transform = data_transform
        self.num_channels = num_channels
        self.loader = loader
        self.train = Training

    def __getitem__(self, index):
        img_paths = self.img_list[index]

        sample = [self.loader(img_paths[i], self.num_channels[i]) for i in range(len(img_paths))]

        # weight = np.array(sample[1])
        # print("weight_max_in_item", np.max(weight))

        if self.train:
            sample = my_aug(sample)
            # sample = torch.cat(sample, sample1)

        if self.data_transform is not None:
            sample = self.data_transform(sample)


        return sample

    def __len__(self):
        return len(self.img_list)


class PairedUnlabeledWithSAMMask(Dataset):
    def __init__(self, filenames, dir_aug, dir_orig, dir_sam_mask, transform_aug=None, transform_orig=None,
                 transform_sam_mask=None):
        self.filenames = filenames  # 直接传入剩余图像名
        self.dir_aug = dir_aug
        self.dir_orig = dir_orig
        self.dir_sam_mask = dir_sam_mask
        self.transform_aug = transform_aug
        self.transform_orig = transform_orig
        self.transform_sam_mask = transform_sam_mask

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # img_aug = Image.open(os.path.join(self.dir_aug, filename)).convert('RGB')
        # img_orig = Image.open(os.path.join(self.dir_orig, filename)).convert('RGB')
        # sam_mask = Image.open(os.path.join(self.dir_sam_mask, filename)).convert('RGB')  # 改成 'L' 如果是灰度

        img_aug = Image.open(os.path.join(self.dir_aug, filename)).convert('L')
        img_orig = Image.open(os.path.join(self.dir_orig, filename)).convert('L')
        sam_mask = Image.open(os.path.join(self.dir_sam_mask, filename)).convert('L')  # 改成 'L' 如果是灰度

        if self.transform_aug:
            img_aug = self.transform_aug([img_aug])[0]
        if self.transform_orig:
            img_orig = self.transform_orig(img_orig)
        if self.transform_sam_mask:
            sam_mask = self.transform_sam_mask([sam_mask])[0]

        return img_aug, img_orig, sam_mask


# class PairedUnlabeledWithSAMMask(Dataset):
#     def __init__(self, dir_aug, dir_orig, dir_sam_mask, transform_aug=None, transform_orig=None,
#                  transform_sam_mask=None):
#         self.dir_aug = dir_aug
#         self.dir_orig = dir_orig
#         self.dir_sam_mask = dir_sam_mask
#         self.transform_aug = transform_aug
#         self.transform_orig = transform_orig
#         self.transform_sam_mask = transform_sam_mask  # 一般只做 ToTensor
#
#         # 假设所有三个目录的文件名完全一致
#         self.filenames = sorted(os.listdir(dir_aug))
#
#     def __len__(self):
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         filename = self.filenames[idx]
#
#         img_aug = Image.open(os.path.join(self.dir_aug, filename)).convert('RGB')
#         img_orig = Image.open(os.path.join(self.dir_orig, filename)).convert('RGB')
#         sam_mask = Image.open(os.path.join(self.dir_sam_mask, filename)).convert('RGB')  # 你如果是灰度图就改成 'L'
#
#         if self.transform_aug:
#             img_aug = self.transform_aug([img_aug])[0]
#         if self.transform_orig:
#             img_orig = self.transform_orig(img_orig)
#         if self.transform_sam_mask:
#             sam_mask = self.transform_sam_mask([sam_mask])[0]
#
#         return img_aug, img_orig, sam_mask

class PairedUnlabeledWithSAMMask_v1(Dataset):
    def __init__(self, dir_aug, dir_orig, dir_sam_mask,
                 transform_aug=None, transform_orig=None, transform_sam_mask=None):
        """
        dir_aug: 增强图像目录
        dir_orig: 原始图像目录
        dir_sam_mask: SAM 生成的 mask 目录
        """
        self.dir_aug = dir_aug
        self.dir_orig = dir_orig
        self.dir_sam_mask = dir_sam_mask
        self.transform_aug = transform_aug
        self.transform_orig = transform_orig
        self.transform_sam_mask = transform_sam_mask

        # 自动读取文件名，假设三个目录下的文件名一致
        self.filenames = sorted(os.listdir(dir_aug))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]



        # 加载图像
        img_aug = Image.open(os.path.join(self.dir_aug, filename)).convert('RGB')
        img_orig = Image.open(os.path.join(self.dir_orig, filename)).convert('RGB')
        sam_mask = Image.open(os.path.join(self.dir_sam_mask, filename)).convert('RGB')  # 如果是灰度改 'L'

        # img_aug = Image.open(os.path.join(self.dir_aug, filename)).convert('L')
        # img_orig = Image.open(os.path.join(self.dir_orig, filename)).convert('L')
        # sam_mask = Image.open(os.path.join(self.dir_sam_mask, filename)).convert('L')  # 如果是灰度改 'L'


        # 应用变换
        if self.transform_aug:
            img_aug = self.transform_aug([img_aug])[0]
        if self.transform_orig:
            img_orig = self.transform_orig(img_orig)
        if self.transform_sam_mask:
            sam_mask = self.transform_sam_mask([sam_mask])[0]


        return img_aug, img_orig, sam_mask


class PairedUnlabeledWithSAMMask_v2(Dataset):
    def __init__(self, dir_aug, dir_orig, dir_sam_mask,
                 transform_aug=None, transform_orig=None, transform_sam_mask=None):
        self.dir_aug = Path(dir_aug)
        self.dir_orig = Path(dir_orig)
        self.dir_sam_mask = Path(dir_sam_mask)
        self.transform_aug = transform_aug
        self.transform_orig = transform_orig
        self.transform_sam_mask = transform_sam_mask

        def list_stems(d: Path):
            exts = {".png", ".jpg", ".jpeg", ".bmp"}
            mp = {}
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in exts:
                    mp[p.stem] = p
            return mp

        aug_map  = list_stems(self.dir_aug)
        orig_map = list_stems(self.dir_orig)
        sam_map  = list_stems(self.dir_sam_mask)

        self.ids = sorted(list(set(aug_map.keys()) & set(orig_map.keys()) & set(sam_map.keys())))
        self.aug_map, self.orig_map, self.sam_map = aug_map, orig_map, sam_map

        print(f"[Paired] aug={len(aug_map)} orig={len(orig_map)} sam={len(sam_map)} intersect={len(self.ids)}")
        if len(self.ids) == 0:
            raise RuntimeError("No matched samples among aug/orig/sam_mask. Check filenames!")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _fix(t):
        if torch.is_tensor(t):
            return t.contiguous().clone()
        return t

    @staticmethod
    def _apply_transform(transform, x):
        """兼容 transform(x) 或 transform([x])[0] 两种写法"""
        if transform is None:
            return x
        y = transform(x)
        if isinstance(y, (list, tuple)):
            return y[0]
        return y

    def __getitem__(self, idx):
        sid = self.ids[idx]

        p_aug  = self.aug_map[sid]
        p_orig = self.orig_map[sid]
        p_sam  = self.sam_map[sid]

        # 读图
        img_aug_pil  = Image.open(p_aug).convert("RGB")
        img_orig_pil = Image.open(p_orig).convert("RGB")

        # sam_mask 如果本质是二值/灰度，建议用 'L'
        sam_pil = Image.open(p_sam).convert("L")   # <- 如果你确定是RGB mask 再改成 "RGB"

        if self.transform_aug is not None:
            img_aug = self.transform_aug([img_aug_pil])[0]
        else:
            img_aug = img_aug_pil

        # ✅ 2) transform_orig：SAMPreprocess 是单图输入
        if self.transform_orig is not None:
            img_orig = self.transform_orig(img_orig_pil)
        else:
            img_orig = img_orig_pil

        # ✅ 3) transform_sam_mask：同样 my_transforms 需要 list 输入（你现在传的是 data_transforms['sam_mask']）
        if self.transform_sam_mask is not None:
            sam_mask = self.transform_sam_mask([sam_pil])[0]
        else:
            sam_mask = sam_pil

        # 最终必须是 Tensor（否则 DataLoader collate 还会炸）
        if not torch.is_tensor(img_aug) or not torch.is_tensor(img_orig) or not torch.is_tensor(sam_mask):
            raise TypeError(f"Transforms must return Tensors. Got: "
                            f"aug={type(img_aug)}, orig={type(img_orig)}, sam={type(sam_mask)} (id={sid})")

        img_aug  = self._fix(img_aug)
        img_orig = self._fix(img_orig)
        sam_mask = self._fix(sam_mask)
 
        return img_aug, img_orig, sam_mask

class PairedUnlabeledDataset(Dataset):
    def __init__(self, dir_aug, dir_orig, transform_aug=None, transform_orig=None):
        self.dir_aug = dir_aug
        self.dir_orig = dir_orig
        self.transform_aug = transform_aug
        self.transform_orig = transform_orig
        self.filenames = sorted(os.listdir(dir_aug))  # 假设两个目录下文件名完全一致

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # img_aug = Image.open(os.path.join(self.dir_aug, filename)).convert('RGB')
        # img_orig = Image.open(os.path.join(self.dir_orig, filename)).convert('RGB')

        img_aug = Image.open(os.path.join(self.dir_aug, filename)).convert('L')
        img_orig = Image.open(os.path.join(self.dir_orig, filename)).convert('L')

        if self.transform_aug:
            img_aug = self.transform_aug([img_aug])[0]  # 注意：传入列表，取第一个
        if self.transform_orig:
            img_orig = self.transform_orig(img_orig)  # 也是列表，取第一个

        return img_aug, img_orig

class SAMPreprocess:
    def __init__(self, output_size=(1024, 1024)):
        self.output_size = output_size

    def __call__(self, img_pil):
        # 将 PIL.Image 转换为 numpy
        img_np = np.array(img_pil)  # shape: (H, W, 3)

        # resize
        img_resized = transform.resize(
            img_np, self.output_size, order=3,
            preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

        # normalize to [0, 1]
        img_norm = (img_resized - img_resized.min()) / np.clip(
            img_resized.max() - img_resized.min(), a_min=1e-8, a_max=None
        )

        # (H, W, 3) -> (3, H, W)
        img_tensor = torch.tensor(img_norm).float().permute(2, 0, 1)

        return img_tensor