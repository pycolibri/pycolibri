import requests
import zipfile
import os 
import numpy as np
import PIL.Image as Image

class CaveDatasetHandler():
    """

    """

    def __init__(self, path: str, download: bool = True, url : str ='https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/complete_ms_data.zip'):
        self.path = path
        self.url = url
        if download:
            self.download()

    def download(self):

        path_file = os.path.join(self.path, 'cave_dataset.zip')
        final_folder_path = path_file.replace('.zip', '')

        if not os.path.exists(final_folder_path):
            r = requests.get(self.url, allow_redirects=True)
            open(path_file, 'wb').write(r.content)

            with zipfile.ZipFile(path_file, 'r') as zip_ref:
                zip_ref.extractall(final_folder_path)
            os.remove(path_file)
        else:
            print('Dataset already downloaded')

    def get_list_paths(self):
        path_files = []
        for name in os.listdir(self.path):
            path_files.append(os.path.join(self.path, name, name))
        return path_files

    def load_item(self, filename: str) -> dict[str, np.ndarray]:

        name = os.path.basename(filename).replace('_ms', '')

        spectral_image = []
        for i in range(1, 32):
            spectral_band_filename = os.path.join(filename, f'{name}_ms_{i:02d}.png')
            spectral_band = np.array(Image.open(spectral_band_filename))
            spectral_band = spectral_band / (2 ** 16 - 1) if isinstance(spectral_band[0, 0], np.uint16) else spectral_band
            spectral_band = spectral_band / (2 ** 8 - 1) if isinstance(spectral_band[0, 0], np.uint8) else spectral_band
            spectral_image.append(spectral_band.astype(np.float32))
        spectral_image = np.stack(spectral_image, axis=-1)

        rgb_image = np.array(Image.open(os.path.join(filename, f'{name}_RGB.bmp'))) / 255.
        rgb_image = rgb_image.astype(np.float32)

        return dict(input=rgb_image, output=spectral_image)

