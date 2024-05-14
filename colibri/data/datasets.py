from torch.utils import data
import colibri.data.utils as D

DATASET_TYPE = {
    'builtin': D.load_builtin_dataset,
    'img': D.load_img_dataset,
    'mat': D.load_mat_dataset,
    'h5': D.load_h5_dataset,
}


class Dataset:
    def __init__(self, path, data_type, preprocessing, transforms, batch_size, use_loader=True, num_workers=0):
        dataset = DATASET_TYPE[data_type]
        train_dataset, test_dataset = dataset(path, preprocessing, transforms)

        if use_loader:
            self.train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers)
            self.test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers)
