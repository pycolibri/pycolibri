import requests
import zipfile
import os 
import numpy as np
import PIL.Image as Image
import torch

class CaveDataset():
    r"""

    Class to handle the CAVE dataset. 

    The CAVE dataset is a database of multispectral images that were used to emulate the GAP camera. The images are of a wide variety of real-world materials and objects.

    URL: https://www.cs.columbia.edu/CAVE/databases/multispectral/
    """

    def __init__(self, path: str, download: bool = True, url : str ='https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip'):
        self.url = url
        self.tmp_name = 'cave_dataset'
        self.path = os.path.join(path, self.tmp_name)
        self.num_channels = 32
        if download:
            self.download()

    def download(self):
        r"""
        Downloads the dataset from the specified URL and extracts it to the specified path.
        """
        
        zip_path = self.path+".zip"
        if not os.path.exists(self.path):
            r = requests.get(self.url, allow_redirects=True)
            open(zip_path, 'wb').write(r.content)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.path)
            os.remove(zip_path)
        else:
            print('Dataset already downloaded')


    def get_list_paths(self):
        r"""
        Returns a list of cave filenames in the given path.

        Args:
            path (str): The path to the directory containing the cave files.
        Returns:
            list: A list of cave filenames.
        """

        path_files = []
        for name in os.listdir(self.path):
            path_files.append(os.path.join(self.path, name, name))
        return path_files

    def load_item(self, filename: str) -> dict:
        r"""
        Load a sample from the CAVE dataset.
        Args:
            filename (str): The filename of the sample.
        Returns:
            dict: A dictionary containing the input and output data of the sample.
        """
        name = os.path.basename(filename).replace('_ms', '')

        spectral_image = []
        for i in range(1, self.num_channels):
            spectral_band_filename = os.path.join(filename, f'{name}_ms_{i:02d}.png')
            spectral_band = np.array(Image.open(spectral_band_filename))
            if len(spectral_band.shape)>2:
                spectral_band = spectral_band[..., :3].mean(axis=-1)
            spectral_band = spectral_band / (2 ** 16 - 1) if isinstance(spectral_band[0, 0], np.uint16) else spectral_band
            spectral_band = spectral_band / (2 ** 8 - 1) if isinstance(spectral_band[0, 0], np.uint8) else spectral_band
            spectral_image.append(spectral_band.astype(np.float32))#[np.newaxis, ...]
        spectral_image = np.stack(spectral_image, axis=0)#[np.newaxis, ...]

        rgb_image = np.array(Image.open(os.path.join(filename, f'{name}_RGB.bmp'))) / 255.
        rgb_image = np.transpose(rgb_image, (2, 0, 1))#[np.newaxis, ...]
        rgb_image = rgb_image.astype(np.float32)

        rgb_image = torch.from_numpy(rgb_image)
        spectral_image = torch.from_numpy(spectral_image)

        return dict(input=rgb_image, output=spectral_image)

