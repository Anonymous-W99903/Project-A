import os
import numpy as np
from torch.utils import data
import torch
import random
import numpy as np
import scipy.io as scio
import os
import tqdm

########################## Widar 3.0 ###############################
DOMAIN_NUM_DICT = {
    "user": [5, 10, 11, 12, 13, 14, 15, 16, 17],  # user
    "ges": 6,  # gesture
    "loc": 5,  # location
    "ori": 5,  # orientation
    "rep": 5,  # repetition
}


# dataset for training the domain-specific teacher model
class DomainSet(data.Dataset):
    def __init__(
        self,
        data_path=None,
        domain_name=None,
        domain_number=None,
        file_lists=None,
        isTest=False,
        preread=True,
        return_domain_label=False,
    ):
        """
        when file_lists is given, other parameters can be omitted
        :param preread: if preread is true, all data will be loaded into memory before the training
        :param isTest: if isTest is true, data augmentation will return a fixed sequence; else return a random sequence
        :param return_domain_label: when it's true, __getitem__ return a triple (data, ges_label, domain_label)

        """
        # get file list
        if file_lists:
            self.file_lists = file_lists
        else:
            assert domain_name in DOMAIN_NUM_DICT
            self.domain_number = domain_number
            filename_lists = get_all_domain_filename(domain_name, domain_number)
            self.file_lists = []
            for f in filename_lists:
                pth = os.path.join(data_path, f)
                if os.path.exists(pth):
                    self.file_lists.append(pth)

        self.data = None
        if preread:
            self.data = [
                scio.loadmat(pth)["doppler_spectrum"]
                for pth in tqdm.tqdm(self.file_lists)
            ]
        self.domain_name = domain_name
        self.isTest = isTest
        self.return_domain_label = return_domain_label

    def __len__(self):
        return len(self.file_lists)

    def __getitem__(self, index):
        # define a function for getting activity groundtruth from filename
        def get_target(x):
            return int(x[x.find("-") + 1 : x.find("-", x.find("-") + 1)])

        def get_domain(x):
            ges_start = x.find("-") + 1
            loc_start = x.find("-", ges_start) + 1
            ori_start = x.find("-", loc_start) + 1
            rep_start = x.find("-", ori_start) + 1
            if self.domain_name == "user":
                return int(x[: ges_start - 1])
            if self.domain_name == "loc":
                return int(x[loc_start : ori_start - 1])
            if self.domain_name == "ori":
                return int(x[ori_start : rep_start - 1])
            raise ValueError("domain name error! error value: %s" % (self.domain_name))

        pth = self.file_lists[index]
        filename = os.path.split(pth)[1]
        if self.data:
            x = self.data[index]
        else:
            x = scio.loadmat(pth)["doppler_spectrum"]
        if x.dtype != np.dtype("float32"):
            x = x.astype(np.float32)
        augmented_x = augment(x, self.isTest)

        # reshape
        augmented_x = np.reshape(augmented_x, (augmented_x.shape[0], 121, 121))
        # to make groundtruth start from 0, substract 1
        if not self.return_domain_label:
            return augmented_x, np.array(get_target(filename)) - 1
        else:
            # return augmented_x, np.array(get_target(filename)) - 1, get_domain(filename)
            return (
                augmented_x,
                np.array(get_target(filename)) - 1,
                get_domain(filename) - 1,
            )


####################### utils #################################
def get_all_domain_filename(domain_name, domain_number):
    """
    :param domain_name:
    :param domain_number: a int or a int list, indicating which domains will be loaded
    """
    print(f"name is {domain_name}, number is {domain_number}")
    domain_number = [domain_number] if isinstance(domain_number, int) else domain_number

    def get_user_range():
        return domain_number if domain_name == "user" else DOMAIN_NUM_DICT["user"]

    def get_range(x):
        return (
            domain_number
            if domain_name == x
            else list(range(1, DOMAIN_NUM_DICT[x] + 1))
        )

    filename_lists = []
    for user in get_user_range():
        for ges in get_range("ges"):
            for loc in get_range("loc"):
                for ori in get_range("ori"):
                    for rep in get_range("rep"):
                        # for d in range(1, 15):
                        #     filename = f'user{user}-{ges}-{loc}-{ori}-{rep}-d{d}.npy'
                        #     filename_lists.append(filename)
                        filename = f"user{user}/user{user}-{ges}-{loc}-{ori}-{rep}.mat"
                        filename_lists.append(filename)
    return filename_lists


def get_train_and_valid_loader(source_domains, args, ratio=0.9, **kwargs):
    print(f"domain_name: {args.domain_name}")
    print(f"source_domains: {source_domains}")
    # define data loader
    assert args.domain_name in DOMAIN_NUM_DICT
    print(f"domain_name is in DOMAIN_NUM_DICT: {args.domain_name in DOMAIN_NUM_DICT}")
    filename_lists = get_all_domain_filename(args.domain_name, source_domains)
    print(f"filename_lists: {len(filename_lists)}")


    file_lists = []
    for f in filename_lists:
        pth = os.path.join(args.data_dir, f)
        # print(args.data_dir)
        # print(f)
        # print(pth)
        if os.path.exists(pth):
            file_lists.append(pth)
    print(f"Number of files in file_lists: {len(file_lists)}")

    # shuffle files
    random.shuffle(file_lists)
    # split into trainset and testset
    boundry = int(len(file_lists) * ratio)
    train_dataset = DomainSet(file_lists=file_lists[:boundry], **kwargs)
    test_dataset = DomainSet(file_lists=file_lists[boundry:], **kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


# def get_train_baseline_loader(args, return_domain_label = False):
#     assert args.domain_name in DOMAIN_NUM_DICT
#
#     def get_path_list(domain_num):
#         filename_lists = get_all_domain_filename(args.domain_name, domain_num)
#         file_lists = []
#         for f in filename_lists :
#             pth = os.path.join(args.data_dir, f)
#             if os.path.exists(pth) :
#                 file_lists.append(pth)
#         return file_lists
#
#     train_dataset = DomainSet(file_lists=get_path_list(args.train_domain), domain_name=args.domain_name, return_domain_label=return_domain_label)
#     test_dataset = DomainSet(file_lists=get_path_list(args.test_domain), domain_name=args.domain_name)
#
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                 batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset,
#                 batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)
#     return train_loader, test_loader


####################### Data  Augmentation #################################
NEW_LEN = 121


def augment(x: np.ndarray, isTest):
    """
    implement data augmentation by downsampling.

    When isTest is False, generate a NEW_LENGTH long sequence A.
    A[i] is a random int in [step*i, step(i+1)), where step = old_len // NEW_LEN.

    When isTest is True, generate  step NEW_LENGTH-long sequences B[0...step-1].
    B[j][i] step*i + j, where step = old_len // NEW_LEN.
    """
    old_len = x.shape[2]
    step = old_len // NEW_LEN
    if step == 0:
        step = 1
    # if not isTest:
    #     index = [random.randint(step * i, step * (i + 1) - 1) for i in range(NEW_LEN)]
    #     new_data = x[:, :, index]
    #     return new_data
    if not isTest:
        index = [min(random.randint(step * i, step * (i + 1) - 1), old_len - 1) for i in range(NEW_LEN)]
        new_data = x[:, :, index]
        # print(f"Augmented shape (train): {new_data.shape}")
        return new_data
    else:
        new_datas = []
        for start in range(step):
            index = list(range(start, old_len, step))[:NEW_LEN]
            new_datas.append(x[:, :, index])
        return new_datas[step // 2]
