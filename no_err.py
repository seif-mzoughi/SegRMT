from jinja2 import Template
import yaml
import copy
import os
import math
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from spectral import imshow, view_cube
import mmcv
import torch 
import warnings
from pathlib import Path
import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
import os
import yaml
import numpy as np
import numpy as np
from skimage import transform
import sys
import copy
from scipy.ndimage import zoom
import copy
import yaml
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from mmseg.apis import init_model
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import os
from mmseg.apis import MMSegInferencer
import matplotlib.pyplot as plt
from skimage.io import imsave
import tifffile
import numpy as np
from PIL import Image
import tifffile
import os
from mmseg.apis import MMSegInferencer
import os
import numpy as np
import pygad  # Import the pygad module
from PIL import Image
import tifffile
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import jaccard_score
import spectral
import gc
np.random.seed(3)
def generate_indices_metadata(params, template_path, shape):
    parameters_metadata = copy.deepcopy(params)
    file_path = "indices_metadata.yaml"  # Save the output in the current working directory

    with open(template_path) as file:
        template = yaml.safe_load(file)

    n_lines, n_columns, n_bands = shape[0], shape[1], shape[2]
    parameters_metadata['n_lines'] = n_lines
    parameters_metadata['n_columns'] = n_columns
    parameters_metadata['n_bands'] = n_bands

    output = {}
    for k, v in template.items():
        output[k] = math.ceil(float(Template(str(v)).render(parameters_metadata)))
        parameters_metadata[k] = output[k]

    with open(file_path, 'w') as file:
        yaml.dump(output, file, default_flow_style=False)

    return output

root_directory = "F:\\mmsegmentation\\cityscapes\\leftImg8bit\\val\\frankfurt"

# List all files in the directory
files = os.listdir(root_directory)

# Assuming there are images in the directory, take the first one
first_image_path = os.path.join(root_directory, files[0])

# Load the image
image = Image.open(first_image_path)

# Convert the PIL Image to a NumPy array
image_array = np.array(image)

# Get the dimensions of the image array
height, width, n_bands = image_array.shape

# Create a shape variable
image_shape = (width, height, n_bands)
shape=image_array.shape
# Print the shape of the image
print("Image shape:", shape)
image_copy = copy.deepcopy(image_array)

class MaxMinGenerator:

    def __init__(self, shape):
        self.n_lines = shape[0]
        self.n_columns = shape[1]
        self.n_bands = shape[2]
        
        self.parameters_metadata = self.build_transformation_metadata("parameters")
        self.indices_metadata = generate_indices_metadata(self.parameters_metadata, 'template.yaml', shape)
    @staticmethod
    def build_transformation_metadata(metadata_type):
        path = os.path.dirname(os.path.abspath(__file__))
        file_name = metadata_type + "_metadata.yaml"
        metadata_path = path +"/"+file_name
        with open(metadata_path) as file:
            tr_metadata = yaml.load(file, Loader=yaml.FullLoader)
        return tr_metadata

    def generate_continuous_line_drop_out_min_max_vector(self):
        vector_size = self.indices_metadata['continuous_line_col_drop_out_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_lines = self.indices_metadata['max_lines'] + 1
        max_bands = self.indices_metadata['max_bands'] + max_lines

        max_vector[np.arange(1, max_lines, 2)] = self.n_lines - 1

        max_vector[np.arange(max_lines, max_bands, 2)] = self.n_bands - 1

        return max_vector, min_vector, var_type_vector

    def generate_discontinuous_line_drop_out_min_max_vector(self):
        vector_size = self.indices_metadata['discontinuous_line_col_drop_out_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_lines = self.indices_metadata['max_lines'] + 1
        max_columns = self.indices_metadata['max_columns'] + max_lines
        max_bands = self.indices_metadata['max_bands'] + max_columns

        max_vector[np.arange(1, max_lines, 2)] = self.n_lines - 1

        max_vector[np.arange(max_lines, max_columns, 2)] = self.n_columns - 1

        max_vector[np.arange(max_columns, max_bands, 2)] = self.n_bands - 1

        return max_vector, min_vector, var_type_vector

    def generate_line_stripping_min_max_vector(self):
        vector_size = self.indices_metadata['line_col_stripping_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_lines = self.indices_metadata['max_lines'] + 1

        max_vector[np.arange(1, max_lines, 2)] = self.n_lines - 1

        max_vector[-2] = self.parameters_metadata["max_mean_bound"] / 100.0
        min_vector[-2] = self.parameters_metadata["min_mean_bound"] / 100.0
        var_type_vector[-2] = 'real'
        max_vector[-1] = self.parameters_metadata["max_std_bound"] / 100.0
        min_vector[-1] = self.parameters_metadata["min_std_bound"] / 100.0
        var_type_vector[-1] = 'real'

        return max_vector, min_vector, var_type_vector

    def generate_line_col_transformations_min_max_vector(self):
        max_transformation_type = 1
        min_transformation_type = 0

        continuous_drop_out_max_vector, continuous_drop_out_min_vector, continuous_drop_out_var_type_vector = \
            self.generate_continuous_line_drop_out_min_max_vector()

        discontinuous_drop_out_max_vector, discontinuous_drop_out_min_vector, discontinuous_drop_out_var_type_vector = \
            self.generate_discontinuous_line_drop_out_min_max_vector()

        stripping_max_vector, stripping_min_vector, stripping_var_type_vector = \
            self.generate_line_stripping_min_max_vector()

        max_vector = np.concatenate((max_transformation_type, continuous_drop_out_max_vector,
                                     discontinuous_drop_out_max_vector, stripping_max_vector), axis=None)
        min_vector = np.concatenate((min_transformation_type, continuous_drop_out_min_vector,
                                     discontinuous_drop_out_min_vector, stripping_min_vector), axis=None)
        var_type_vector = np.concatenate(("int", continuous_drop_out_var_type_vector,
                                          discontinuous_drop_out_var_type_vector, stripping_var_type_vector), axis=None)

        return max_vector, min_vector, var_type_vector

    def generate_region_drop_out_min_max_vector(self):
        vector_size = self.indices_metadata['region_drop_out_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_bands = self.indices_metadata['max_bands'] + 1
        max_x, max_width = self.n_lines - 1, self.n_lines - 1
        max_y, max_length = self.n_columns - 1, self.n_columns - 1

        max_vector[np.arange(1, max_bands, 2)] = self.n_bands - 1

        max_vector[-5:-1] = max_x, max_y, max_width, max_length

        return max_vector, min_vector, var_type_vector

    def generate_spectral_band_loss_min_max_vector(self):
        vector_size = self.indices_metadata['spectral_band_loss_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_bands = self.indices_metadata['max_bands_for_sbl'] + 1

        max_vector[np.arange(1, max_bands, 2)] = self.n_bands - 2
        min_vector[np.arange(1, max_bands, 2)] = 1

        return max_vector, min_vector, var_type_vector

    def generate_salt_and_pepper_min_max_vector(self):
        vector_size = self.indices_metadata['salt_pepper_noise_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_noisy_pixels = int(np.ceil((self.parameters_metadata[
                                            'max_percentage_of_salt_pepper_noisy_pixels'] *
                                        self.n_lines * self.n_columns) / 100) * 3 + 1)

        max_vector[np.arange(1, max_noisy_pixels, 3)] = self.n_lines - 1
        max_vector[np.arange(2, max_noisy_pixels, 3)] = self.n_columns - 1
        max_vector[np.arange(3, max_noisy_pixels, 3)] = 0
        min_vector[np.arange(3, max_noisy_pixels, 3)] = -2

        return max_vector, min_vector, var_type_vector

    def generate_spatial_gaussian_noise_min_max_vector(self):
        vector_size = self.indices_metadata['spatial_gaussian_noise_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_vector[np.arange(1, vector_size, 3)] = self.n_lines - 1
        max_vector[np.arange(2, vector_size, 3)] = self.n_columns - 1
        max_vector[np.arange(3, vector_size, 3)] = 1
        var_type_vector[np.arange(3, vector_size, 3)] = "real"

        return max_vector, min_vector, var_type_vector

    def generate_spectral_gaussian_noise_min_max_vector(self):
        vector_size = self.indices_metadata['spectral_gaussian_noise_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_vector[np.arange(1, vector_size, 2)] = self.n_bands - 1
        max_vector[np.arange(2, vector_size, 2)] = 1
        var_type_vector[np.arange(2, vector_size, 2)] = "real"

        return max_vector, min_vector, var_type_vector

    def generate_rotation_min_max_vector(self):
        vector_size = self.indices_metadata['rotation_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        x, y = self.patch_index
        max_angle = self.parameters_metadata['max_rotation_angle']
        min_angle = self.parameters_metadata['min_rotation_angle']

        max_vector[1:] = x, y, max_angle
        min_vector[1:] = x, y, min_angle
        var_type_vector[-1] = "real"

        return max_vector, min_vector, var_type_vector
        
    
    def generate_zoom_min_max_vector(self):
        vector_size = self.indices_metadata['zoom_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        x, y = self.patch_index
        max_zoom_factor = self.parameters_metadata['max_zoom_factor']
        min_zoom_factor = self.parameters_metadata['min_zoom_factor']

        max_vector[1:] = x, y, max_zoom_factor
        min_vector[1:] = x, y, min_zoom_factor
        var_type_vector[-1] = 'real'

        return max_vector, min_vector, var_type_vector
    
    def generate_max_min_vector(self):
        vector_size = self.indices_metadata['vector_size']

        line_col_transformation_max_vector, line_col_transformation_min_vector, line_col_transformation_var_type_vector \
            = self.generate_line_col_transformations_min_max_vector()
        region_drop_out_max_vector, region_drop_out_min_vector, region_drop_out_var_type_vector = \
            self.generate_region_drop_out_min_max_vector()
        spectral_band_loss_max_vector, spectral_band_loss_min_vector, spectral_band_loss_var_type_vector = \
            self.generate_spectral_band_loss_min_max_vector()
        salt_pepper_noise_max_vector, salt_pepper_noise_min_vector, salt_pepper_noise_var_type_vector = \
            self.generate_salt_and_pepper_min_max_vector()
        spatial_gaussian_noise_max_vector, spatial_gaussian_noise_min_vector, spatial_gaussian_noise_var_type_vector = \
            self.generate_spatial_gaussian_noise_min_max_vector()
        spectral_gn_max_vector, spectral_gn_min_vector, spectral_gn_var_type_vector = \
            self.generate_spectral_gaussian_noise_min_max_vector()
        rotation_max_vector, rotation_min_vector, rotation_var_type_vector = self.generate_rotation_min_max_vector()
        zoom_max_vector, zoom_min_vector, zoom_var_type_vector = self.generate_zoom_min_max_vector()

        final_max_vector = np.concatenate((line_col_transformation_max_vector,
                                           region_drop_out_max_vector,
                                           spectral_band_loss_max_vector,
                                           salt_pepper_noise_max_vector,
                                           spatial_gaussian_noise_max_vector,
                                           spectral_gn_max_vector,
                                           rotation_max_vector,
                                           zoom_max_vector), axis=None)

        final_min_vector = np.concatenate((line_col_transformation_min_vector,
                                           region_drop_out_min_vector,
                                           spectral_band_loss_min_vector,
                                           salt_pepper_noise_min_vector,
                                           spatial_gaussian_noise_min_vector,
                                           spectral_gn_min_vector,
                                           rotation_min_vector,
                                           zoom_min_vector), axis=None)

        final_var_type_vector = np.concatenate((line_col_transformation_var_type_vector,
                                                region_drop_out_var_type_vector,
                                                spectral_band_loss_var_type_vector,
                                                salt_pepper_noise_var_type_vector,
                                                spatial_gaussian_noise_var_type_vector,
                                                spectral_gn_var_type_vector,
                                                rotation_var_type_vector,
                                                zoom_var_type_vector), axis=None)

        assert (vector_size == final_max_vector.shape[0]) and (vector_size == final_min_vector.shape[0]), \
            "the shape of the final max or min transformation vector is wrong !!!!"

        return final_max_vector, final_min_vector, final_var_type_vector

    def format_max_min_vector(self):
        min_vector, max_vector, var_type_vector = self.generate_max_min_vector()
        max_min_vector = np.dstack((max_vector, min_vector))

        return max_min_vector.reshape((max_min_vector.shape[1], max_min_vector.shape[2])), var_type_vector.reshape(
            (var_type_vector.shape[0], 1))
        
class MaxMinGenerator:
    

    def __init__(self, shape):
        self.n_lines = shape[0]
        self.n_columns = shape[1]
        self.n_bands = shape[2]
        params_path_file='F:\\mmsegmentation\\parameters_metadata.yaml'
        self.parameters_metadata = self.build_transformation_metadata(params_path_file)
        self.indices_metadata = generate_indices_metadata(self.parameters_metadata, 'template.yaml', shape)
    params_path_file='F:\\mmsegmentation\\parameters_metadata.yaml'
    

    @staticmethod
    def build_transformation_metadata(params_file_path):
        with open(params_file_path) as file:
            tr_metadata = yaml.load(file, Loader=yaml.FullLoader)
        return tr_metadata

    def generate_continuous_line_drop_out_min_max_vector(self):
        vector_size = self.indices_metadata['continuous_line_col_drop_out_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_lines = self.indices_metadata['max_lines'] + 1
        max_bands = self.indices_metadata['max_bands'] + max_lines

        max_vector[np.arange(1, max_lines, 2)] = self.n_lines - 1

        max_vector[np.arange(max_lines, max_bands, 2)] = self.n_bands - 1

        return max_vector, min_vector, var_type_vector

    def generate_discontinuous_line_drop_out_min_max_vector(self):
        vector_size = self.indices_metadata['discontinuous_line_col_drop_out_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_lines = self.indices_metadata['max_lines'] + 1
        max_columns = self.indices_metadata['max_columns'] + max_lines
        max_bands = self.indices_metadata['max_bands'] + max_columns

        max_vector[np.arange(1, max_lines, 2)] = self.n_lines - 1

        max_vector[np.arange(max_lines, max_columns, 2)] = self.n_columns - 1

        max_vector[np.arange(max_columns, max_bands, 2)] = self.n_bands - 1

        return max_vector, min_vector, var_type_vector

    def generate_line_stripping_min_max_vector(self):
        vector_size = self.indices_metadata['line_col_stripping_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_lines = self.indices_metadata['max_lines'] + 1

        max_vector[np.arange(1, max_lines, 2)] = self.n_lines - 1

        max_vector[-2] = self.parameters_metadata["max_mean_bound"] / 100.0
        min_vector[-2] = self.parameters_metadata["min_mean_bound"] / 100.0
        var_type_vector[-2] = 'real'
        max_vector[-1] = self.parameters_metadata["max_std_bound"] / 100.0
        min_vector[-1] = self.parameters_metadata["min_std_bound"] / 100.0
        var_type_vector[-1] = 'real'

        return max_vector, min_vector, var_type_vector

    def generate_line_col_transformations_min_max_vector(self):
        max_transformation_type = 1
        min_transformation_type = 0

        continuous_drop_out_max_vector, continuous_drop_out_min_vector, continuous_drop_out_var_type_vector = \
            self.generate_continuous_line_drop_out_min_max_vector()

        discontinuous_drop_out_max_vector, discontinuous_drop_out_min_vector, discontinuous_drop_out_var_type_vector = \
            self.generate_discontinuous_line_drop_out_min_max_vector()

        stripping_max_vector, stripping_min_vector, stripping_var_type_vector = \
            self.generate_line_stripping_min_max_vector()

        max_vector = np.concatenate((max_transformation_type, continuous_drop_out_max_vector,
                                     discontinuous_drop_out_max_vector, stripping_max_vector), axis=None)
        min_vector = np.concatenate((min_transformation_type, continuous_drop_out_min_vector,
                                     discontinuous_drop_out_min_vector, stripping_min_vector), axis=None)
        var_type_vector = np.concatenate(("int", continuous_drop_out_var_type_vector,
                                          discontinuous_drop_out_var_type_vector, stripping_var_type_vector), axis=None)

        return max_vector, min_vector, var_type_vector

    def generate_region_drop_out_min_max_vector(self):
        vector_size = self.indices_metadata['region_drop_out_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_bands = self.indices_metadata['max_bands'] + 1
        max_x, max_width = self.n_lines - 1, self.n_lines - 1
        max_y, max_length = self.n_columns - 1, self.n_columns - 1

        max_vector[np.arange(1, max_bands, 2)] = self.n_bands - 1

        max_vector[-5:-1] = max_x, max_y, max_width, max_length

        return max_vector, min_vector, var_type_vector

    def generate_spectral_band_loss_min_max_vector(self):
        vector_size = self.indices_metadata['spectral_band_loss_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_bands = self.indices_metadata['max_bands_for_sbl'] + 1

        max_vector[np.arange(1, max_bands, 2)] = self.n_bands - 2
        min_vector[np.arange(1, max_bands, 2)] = 1

        return max_vector, min_vector, var_type_vector

    def generate_salt_and_pepper_min_max_vector(self):
        vector_size = self.indices_metadata['salt_pepper_noise_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_noisy_pixels = int(np.ceil((self.parameters_metadata[
                                            'max_percentage_of_salt_pepper_noisy_pixels'] *
                                        self.n_lines * self.n_columns) / 100) * 3 + 1)

        max_vector[np.arange(1, max_noisy_pixels, 3)] = self.n_lines - 1
        max_vector[np.arange(2, max_noisy_pixels, 3)] = self.n_columns - 1
        max_vector[np.arange(3, max_noisy_pixels, 3)] = 0
        min_vector[np.arange(3, max_noisy_pixels, 3)] = -2

        return max_vector, min_vector, var_type_vector

    def generate_spatial_gaussian_noise_min_max_vector(self):
        vector_size = self.indices_metadata['spatial_gaussian_noise_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_vector[np.arange(1, vector_size, 3)] = self.n_lines - 1
        max_vector[np.arange(2, vector_size, 3)] = self.n_columns - 1
        max_vector[np.arange(3, vector_size, 3)] = 1
        var_type_vector[np.arange(3, vector_size, 3)] = "real"

        return max_vector, min_vector, var_type_vector

    def generate_spectral_gaussian_noise_min_max_vector(self):
        vector_size = self.indices_metadata['spectral_gaussian_noise_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_vector[np.arange(1, vector_size, 2)] = self.n_bands - 1
        max_vector[np.arange(2, vector_size, 2)] = 1
        var_type_vector[np.arange(2, vector_size, 2)] = "real"

        return max_vector, min_vector, var_type_vector

    def generate_rotation_min_max_vector(self):
        vector_size = self.indices_metadata['rotation_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_angle = self.parameters_metadata['max_rotation_angle']
        min_angle = self.parameters_metadata['min_rotation_angle']

        max_vector[1:] = max_angle
        min_vector[1:] = min_angle
        var_type_vector[-1] = "real"

        return max_vector, min_vector, var_type_vector

    def generate_zoom_min_max_vector(self):
        vector_size = self.indices_metadata['zoom_size']
        max_vector = np.ones(shape=(vector_size,))
        min_vector = np.zeros(shape=(vector_size,))
        var_type_vector = np.full(shape=(vector_size,), fill_value="int", dtype="S4")

        max_zoom_factor = self.parameters_metadata['max_zoom_factor']
        min_zoom_factor = self.parameters_metadata['min_zoom_factor']

        max_vector[1:] = max_zoom_factor
        min_vector[1:] = min_zoom_factor
        var_type_vector[-1] = 'real'

        return max_vector, min_vector, var_type_vector

    def generate_max_min_vector(self):
        vector_size = self.indices_metadata['vector_size']

        line_col_transformation_max_vector, line_col_transformation_min_vector, line_col_transformation_var_type_vector \
            = self.generate_line_col_transformations_min_max_vector()
        region_drop_out_max_vector, region_drop_out_min_vector, region_drop_out_var_type_vector = \
            self.generate_region_drop_out_min_max_vector()
        spectral_band_loss_max_vector, spectral_band_loss_min_vector, spectral_band_loss_var_type_vector = \
            self.generate_spectral_band_loss_min_max_vector()
        salt_pepper_noise_max_vector, salt_pepper_noise_min_vector, salt_pepper_noise_var_type_vector = \
            self.generate_salt_and_pepper_min_max_vector()
        spatial_gaussian_noise_max_vector, spatial_gaussian_noise_min_vector, spatial_gaussian_noise_var_type_vector = \
            self.generate_spatial_gaussian_noise_min_max_vector()
        spectral_gn_max_vector, spectral_gn_min_vector, spectral_gn_var_type_vector = \
            self.generate_spectral_gaussian_noise_min_max_vector()
        rotation_max_vector, rotation_min_vector, rotation_var_type_vector = self.generate_rotation_min_max_vector()
        zoom_max_vector, zoom_min_vector, zoom_var_type_vector = self.generate_zoom_min_max_vector()

        final_max_vector = np.concatenate((line_col_transformation_max_vector,
                                           region_drop_out_max_vector,
                                           spectral_band_loss_max_vector,
                                           salt_pepper_noise_max_vector,
                                           spatial_gaussian_noise_max_vector,
                                           spectral_gn_max_vector,
                                           rotation_max_vector,
                                           zoom_max_vector), axis=None)

        final_min_vector = np.concatenate((line_col_transformation_min_vector,
                                           region_drop_out_min_vector,
                                           spectral_band_loss_min_vector,
                                           salt_pepper_noise_min_vector,
                                           spatial_gaussian_noise_min_vector,
                                           spectral_gn_min_vector,
                                           rotation_min_vector,
                                           zoom_min_vector), axis=None)

        final_var_type_vector = np.concatenate((line_col_transformation_var_type_vector,
                                                region_drop_out_var_type_vector,
                                                spectral_band_loss_var_type_vector,
                                                salt_pepper_noise_var_type_vector,
                                                spatial_gaussian_noise_var_type_vector,
                                                spectral_gn_var_type_vector,
                                                rotation_var_type_vector,
                                                zoom_var_type_vector), axis=None)

        assert (vector_size == final_max_vector.shape[0]) and (vector_size == final_min_vector.shape[0]), \
            "the shape of the final max or min transformation vector is wrong !!!!"

        return final_max_vector, final_min_vector, final_var_type_vector

    def format_max_min_vector(self):
        min_vector, max_vector, var_type_vector = self.generate_max_min_vector()
        max_min_vector = np.dstack((max_vector, min_vector))

        return max_min_vector.reshape((max_min_vector.shape[1], max_min_vector.shape[2])), var_type_vector.reshape(
            (var_type_vector.shape[0], 1))
        
params_path = "parameters_metadata.yaml"

template_path = "template.yaml"
output_path = "indices_metadata.yaml"
with open(params_path) as file:
    params = yaml.safe_load(file)

output = generate_indices_metadata(params, template_path, shape)  # Unpack the shape tuple
max_min_generator = MaxMinGenerator(shape)


continuous_line_max_vector, continuous_line_min_vector, continuous_line_var_type_vector = \
    max_min_generator.generate_continuous_line_drop_out_min_max_vector()

continuous_line_max_vector, continuous_line_min_vector, continuous_line_var_type_vector = \
    max_min_generator.generate_continuous_line_drop_out_min_max_vector()


def generate_activation_value():
    return np.random.randint(low=0, high=2)


def generate_random_percentage(max_percentage=100.0, min_percentage=0.0):
    percentage = np.random.uniform(min_percentage / 100.0, max_percentage / 100.0)
    return percentage


def generate_noise_indices(max_elements, number_noisy):
    indices = np.random.choice(max_elements, size=number_noisy, replace=False)
    return indices

"""
def generate_activated_noisy_elements(number_elements, max_percentage, min_percentage):
    number_noisy_lines = math.ceil(number_elements * generate_random_percentage(
        max_percentage=max_percentage,
        min_percentage=min_percentage))
    activated_lines_indices = generate_noise_indices(max_elements=number_elements,
                                                     number_noisy=number_noisy_lines)
    return number_noisy_lines, activated_lines_indices

"""

def generate_activated_noisy_elements(number_elements, max_percentage, min_percentage):
    
    #print(f"Generating noisy elements: number_elements={number_elements}, max_percentage={max_percentage}, min_percentage={min_percentage}")
    
    # Ensure max_percentage and min_percentage are within valid range [0, 1]
    

    number_noisy_lines = math.ceil(number_elements * generate_random_percentage(
        max_percentage=max_percentage,
        min_percentage=min_percentage))
    
    #print(f"number_noisy_lines: {number_noisy_lines}")

    activated_lines_indices = generate_noise_indices(max_elements=number_elements,
                                                     number_noisy=number_noisy_lines)
    
    #print(f"activated_lines_indices: {activated_lines_indices}")

    return number_noisy_lines, activated_lines_indices



def generate_random_binary_values():
    return np.random.choice([0, 1])


def get_random_salt_pepper_indices(pepper_xs, pepper_ys, salt_xs, salt_ys, max_noisy_pixels):
    selected_indices = np.random.choice(np.arange(len(pepper_xs) + len(salt_xs)), int(max_noisy_pixels),
                                        replace=False)

    pepper_indices = selected_indices[selected_indices < len(pepper_xs)]
    salt_indices = selected_indices[selected_indices >= len(pepper_xs)] - len(pepper_xs)
    pepper_xs = pepper_xs[pepper_indices]
    pepper_ys = pepper_ys[pepper_indices]
    salt_xs = salt_xs[salt_indices]
    salt_ys = salt_ys[salt_indices]

    return pepper_xs, pepper_ys, salt_xs, salt_ys


class VectorEncoder(object):
    


    
    def __init__(self, shape):
        params_path_file='F:\\mmsegmentation\\parameters_metadata.yaml'
        self.n_lines = shape[0]
        self.n_columns = shape[1]
        self.n_bands = shape[2]
        
        self.parameters_metadata = self.build_transformation_metadata(params_path_file)
        self.indices_metadata = generate_indices_metadata(self.parameters_metadata, "template.yaml", shape)
        
    def build_transformation_metadata(self,params_file_path):
        with open(params_file_path) as file:
            tr_metadata = yaml.load(file, Loader=yaml.FullLoader)
        return tr_metadata

    def generate_continuous_line_col_drop_out_vector(self):
        vector_size = self.indices_metadata['continuous_line_col_drop_out_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_lines, activated_lines_indices = generate_activated_noisy_elements(
            number_elements=self.n_lines,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_lines"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_lines"])

        number_noisy_bands, activated_bands_indices = generate_activated_noisy_elements(
            number_elements=self.n_bands,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_bands"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands"])

        pixel_type = generate_random_binary_values()
        activate_trf = generate_random_binary_values()

        max_lines = self.indices_metadata['max_lines'] + 1

        output_vector[np.arange(1, number_noisy_lines * 2 + 1, 2)] = activated_lines_indices
        output_vector[np.arange(2, number_noisy_lines * 2 + 1, 2)] = 1

        output_vector[np.arange(max_lines, max_lines + number_noisy_bands * 2, 2)] = activated_bands_indices
        output_vector[np.arange(max_lines + 1, max_lines + number_noisy_bands * 2, 2)] = 1

        output_vector[0] = activate_trf
        output_vector[-1] = pixel_type

        return output_vector

    def generate_discontinuous_line_drop_out_vector(self):
        vector_size = self.indices_metadata['discontinuous_line_col_drop_out_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_lines, activated_lines_indices = generate_activated_noisy_elements(
            number_elements=self.n_lines,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_lines"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_lines"])

        number_noisy_columns, activated_columns_indices = generate_activated_noisy_elements(
            number_elements=self.n_lines,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_columns"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_columns"])

        number_noisy_bands, activated_bands_indices = generate_activated_noisy_elements(
            number_elements=self.n_bands,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_bands"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands"])

        pixel_type = generate_random_binary_values()
        activate_trf = generate_random_binary_values()

        max_lines = self.indices_metadata['max_lines'] + 1
        max_lines_columns = self.indices_metadata['max_columns'] + max_lines

        output_vector[np.arange(1, number_noisy_lines * 2 + 1, 2)] = activated_lines_indices
        output_vector[np.arange(2, number_noisy_lines * 2 + 1, 2)] = 1

        output_vector[np.arange(max_lines, max_lines + number_noisy_columns * 2, 2)] = activated_columns_indices
        output_vector[np.arange(max_lines + 1, max_lines + number_noisy_columns * 2, 2)] = 1

        output_vector[np.arange(max_lines_columns, max_lines_columns + number_noisy_bands * 2, 2)] = \
            activated_bands_indices
        output_vector[np.arange(max_lines_columns + 1, max_lines_columns + number_noisy_bands * 2, 2)] = 1

        output_vector[0] = activate_trf
        output_vector[-1] = pixel_type

        return output_vector

    def generate_line_stripping_vector(self):
        vector_size = self.indices_metadata['line_col_stripping_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_lines, activated_lines_indices = generate_activated_noisy_elements(
            number_elements=self.n_lines,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_lines"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_lines"])

        mean = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_mean_bound"],
            min_percentage=self.parameters_metadata["min_mean_bound"])
        std = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_std_bound"],
            min_percentage=self.parameters_metadata["min_std_bound"])

        activate_trf = generate_random_binary_values()

        output_vector[np.arange(1, number_noisy_lines * 2 + 1, 2)] = activated_lines_indices
        output_vector[np.arange(2, number_noisy_lines * 2 + 1, 2)] = 1

        output_vector[0] = activate_trf
        output_vector[-2] = mean
        output_vector[-1] = std

        return output_vector

    def generate_line_col_transformations_vector(self):
        transformation_type = generate_random_binary_values()
        continuous_drop_out_vector = self.generate_continuous_line_col_drop_out_vector()
        discontinuous_drop_out_vector = self.generate_discontinuous_line_drop_out_vector()
        stripping_vector = self.generate_line_stripping_vector()

        return np.concatenate((transformation_type, continuous_drop_out_vector, discontinuous_drop_out_vector,
                               stripping_vector), axis=None)

    def generate_region_drop_out_vector(self):
        vector_size = self.indices_metadata['region_drop_out_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_bands, activated_bands_indices = generate_activated_noisy_elements(
            number_elements=self.n_bands,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_bands"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands"])

        x, y = (np.random.randint(low=0, high=self.n_lines), np.random.randint(low=0, high=self.n_columns))
        width = np.random.randint(low=0, high=self.n_lines - x)
        length = np.random.randint(low=0, high=self.n_columns - y)
        pixel_type = generate_random_binary_values()
        activate_trf = generate_random_binary_values()

        output_vector[np.arange(1, number_noisy_bands * 2 + 1, 2)] = activated_bands_indices
        output_vector[np.arange(2, number_noisy_bands * 2 + 1, 2)] = 1

        output_vector[0] = activate_trf
        output_vector[-5:] = x, y, width, length, pixel_type

        return output_vector

    def generate_spectral_band_loss_vector(self):
        vector_size = self.indices_metadata['spectral_band_loss_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_bands, activated_bands_indices = generate_activated_noisy_elements(
            number_elements=self.n_bands,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_bands_for_sbl"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands_for_sbl"])

        activated_bands_indices = np.where(activated_bands_indices == 0, 1, activated_bands_indices)
        activated_bands_indices = np.where(activated_bands_indices == self.n_bands - 1, self.n_bands - 2,
                                           activated_bands_indices)
        activate_trf = generate_random_binary_values()

        output_vector[np.arange(1, number_noisy_bands * 2 + 1, 2)] = activated_bands_indices
        output_vector[np.arange(2, number_noisy_bands * 2 + 1, 2)] = 1
        output_vector[0] = activate_trf

        return output_vector

    def generate_salt_and_pepper_vector(self):
        vector_size = self.indices_metadata['salt_pepper_noise_size']
        output_vector = np.zeros(shape=(vector_size,))

        s_vs_p = generate_random_percentage()
        amount = generate_random_percentage(
            min_percentage=self.parameters_metadata['min_percentage_of_salt_pepper_noisy_pixels'],
            max_percentage=self.parameters_metadata['max_percentage_of_salt_pepper_noisy_pixels'])

        # this will maybe generate more than 4 salt and pepper pixels
        mask = np.random.choice([-1, 0, -2], (self.n_lines, self.n_columns),
                                p=[s_vs_p * amount, 1 - (s_vs_p * amount + (1 - s_vs_p) * amount),
                                   (1 - s_vs_p) * amount])

        pepper_xs, pepper_ys = np.where(mask == -1)
        salt_xs, salt_ys = np.where(mask == -2)

        activate_trf = generate_random_binary_values()

        max_noisy_pixels = np.ceil((self.parameters_metadata[
                               'max_percentage_of_salt_pepper_noisy_pixels'] * self.n_lines * self.n_columns) / 100)
        if (len(pepper_xs) + len(salt_xs)) > max_noisy_pixels:
            pepper_xs, pepper_ys, salt_xs, salt_ys = get_random_salt_pepper_indices(pepper_xs, pepper_ys, salt_xs,
                                                                                    salt_ys, max_noisy_pixels)

        output_vector[np.arange(1, len(pepper_xs) * 3 + 1, 3)] = pepper_xs
        output_vector[np.arange(2, len(pepper_ys) * 3 + 1, 3)] = pepper_ys
        output_vector[np.arange(3, len(pepper_ys) * 3 + 1, 3)] = -1

        output_vector[np.arange(len(pepper_ys) * 3 + 1, len(pepper_ys) * 3 + 1 + len(salt_xs) * 3, 3)] = salt_xs
        output_vector[np.arange(len(pepper_xs) * 3 + 2, len(pepper_xs) * 3 + 1 + len(salt_ys) * 3, 3)] = salt_ys
        output_vector[np.arange(len(pepper_xs) * 3 + 3, len(pepper_xs) * 3 + 1 + len(salt_ys) * 3, 3)] = -2

        output_vector[0] = activate_trf

        return output_vector

    def generate_spatial_gaussian_noise_vector(self):
        vector_size = self.indices_metadata['spatial_gaussian_noise_size']
        output_vector = np.zeros(shape=(vector_size,))

        gaussian_p = generate_random_percentage(
            min_percentage=self.parameters_metadata['min_percentage_of_gaussian_noisy_pixels'],
            max_percentage=self.parameters_metadata['max_percentage_of_gaussian_noisy_pixels'])

        mean = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_mean_bound"],
            min_percentage=self.parameters_metadata["min_mean_bound"])
        sigma = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_std_bound"],
            min_percentage=self.parameters_metadata["min_std_bound"])

        mask = np.random.choice([1, 0], (self.n_lines, self.n_columns), p=[gaussian_p, 1 - gaussian_p]) \
            * np.random.normal(mean, sigma, (self.n_lines, self.n_columns))

        gaussian_xs, gaussian_ys = np.where(mask != 0)

        activate_trf = generate_random_binary_values()
        output_vector[0] = activate_trf

        max_noisy_pixels = self.indices_metadata['max_pixels_for_spatial_gn'] / 3

        if len(gaussian_xs) > max_noisy_pixels:
            selected_indices = np.random.choice(np.arange(len(gaussian_xs)), int(max_noisy_pixels),
                                                replace=False)
            gaussian_xs = gaussian_xs[selected_indices]
            gaussian_ys = gaussian_ys[selected_indices]

        output_vector[np.arange(1, len(gaussian_xs) * 3 + 1, 3)] = gaussian_xs
        output_vector[np.arange(2, len(gaussian_xs) * 3 + 1, 3)] = gaussian_ys
        output_vector[np.arange(3, len(gaussian_xs) * 3 + 1, 3)] = mask[gaussian_xs, gaussian_ys]

        return output_vector

    def generate_spectral_gaussian_noise_vector(self):
        vector_size = self.indices_metadata['spectral_gaussian_noise_size']
        output_vector = np.zeros(shape=(vector_size,))

        number_noisy_bands, activated_bands_indices = generate_activated_noisy_elements(
            number_elements=self.n_bands,
            max_percentage=self.parameters_metadata["max_percentage_of_distorted_bands_for_gn"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands_for_gn"])
        
        #print("Generated activated bands indices:", activated_bands_indices)

        mean = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_mean_bound"],
            min_percentage=self.parameters_metadata["min_mean_bound"])
        sigma = generate_random_percentage(
            max_percentage=self.parameters_metadata["max_std_bound"],
            min_percentage=self.parameters_metadata["min_percentage_of_distorted_bands_for_gn"])

        gaussian_noise_values = np.random.normal(mean, sigma, number_noisy_bands)
        activate_trf = generate_random_binary_values()

        output_vector[np.arange(1, number_noisy_bands * 2 + 1, 2)] = activated_bands_indices
        output_vector[np.arange(2, number_noisy_bands * 2 + 1, 2)] = gaussian_noise_values
        output_vector[0] = activate_trf

        #print("Generated spectral gaussian noise vector:", output_vector)
        return output_vector

    def generate_rotation_vector(self):
        vector_size = self.indices_metadata['rotation_size']
        output_vector = np.zeros(shape=(vector_size,))

        angle = float("{:.3f}".format(np.random.uniform(self.parameters_metadata['min_rotation_angle'],
                                                    self.parameters_metadata['max_rotation_angle'])))
        if angle % 90 == 0:
            angle += np.random.uniform(0.1, 1.0)

        activate_trf = generate_random_binary_values()

        output_vector[0] = activate_trf
        output_vector[1] = angle

        return output_vector

    def generate_zoom_vector(self):
        vector_size = self.indices_metadata['zoom_size']
        output_vector = np.zeros(shape=(vector_size,))

        zoom_factor = float("{:.3f}".format(np.random.uniform(self.parameters_metadata['min_zoom_factor'],
                                                          self.parameters_metadata['max_zoom_factor'])))
        if zoom_factor == 1:
            zoom_factor += np.random.uniform(0.1, 1.0)

        activate_trf = generate_random_binary_values()

        output_vector[0] = activate_trf
        output_vector[1] = zoom_factor

        return output_vector

    def construct_random_tr_vector(self):
        vector_size = self.indices_metadata['vector_size']
        

        line_col_transformation_vector = self.generate_line_col_transformations_vector()
        region_drop_out_vector = self.generate_region_drop_out_vector()
        spectral_band_loss_vector = self.generate_spectral_band_loss_vector()
        salt_pepper_noise_vector = self.generate_salt_and_pepper_vector()
        spatial_gaussian_noise_vector = self.generate_spatial_gaussian_noise_vector()
        spectral_gaussian_noise_vector = self.generate_spectral_gaussian_noise_vector()
        
        print(spectral_gaussian_noise_vector)
        rotation_vector = self.generate_rotation_vector()
        zoom_vector = self.generate_zoom_vector()
        #print("Line Col Transformation Vector:", line_col_transformation_vector.shape)
        #print("Region Drop Out Vector:", region_drop_out_vector.shape)
        #print("Spectral Band Loss Vector:", spectral_band_loss_vector.shape)
        #print("Salt and Pepper Vector:", salt_pepper_noise_vector.shape)
        #print("Spatial Gaussian Noise Vector:", spatial_gaussian_noise_vector.shape)
        #print("Spectral Gaussian Noise Vector:", spectral_gaussian_noise_vector.shape)
        #print("Rotation Vector:", rotation_vector.shape)
        #print("Zoom Vector:", zoom_vector.shape)

        final_vector = np.concatenate((line_col_transformation_vector,
                                       region_drop_out_vector,
                                       spectral_band_loss_vector,
                                       salt_pepper_noise_vector,
                                       spatial_gaussian_noise_vector,
                                       spectral_gaussian_noise_vector,
                                       rotation_vector,
                                       zoom_vector), axis=None)

        assert vector_size == final_vector.shape[0], \
            "the shape of the final transformation vector is wrong !!!!"

        return final_vector

encoder=VectorEncoder(shape)
vect=encoder.generate_spectral_gaussian_noise_vector()

def select_neighboring_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


class Transformer(object):

    def __init__(self, whole_data):
        self.whole_data = whole_data
        self.whole_data_copy = copy.deepcopy(whole_data)

    def set_whole_data(self, whole_data):
        self.whole_data = whole_data

    @staticmethod
    def continuous_line_drop_out(whole_data, noisy_lines_indices, noisy_bands_indices, pixels_values):
        #print("************** applying continuous line dropout ***************************")

        # Ensure pixels_values is not empty and properly initialized
        if pixels_values.size == 0:
            raise ValueError("pixels_values is empty")

        pixels_values = np.asarray(pixels_values)
        #print("pixels_values shape:", pixels_values.shape)

        # Ensure pixels_values is at least a 2D array for the next steps
        if pixels_values.ndim == 0:
            pixels_values = np.expand_dims(pixels_values, axis=(0, 1))
        elif pixels_values.ndim == 1:
            pixels_values = np.expand_dims(pixels_values, axis=(0, 1))

        #print("reshaped pixels_values shape:", pixels_values.shape)
        #print("noisy_lines_indices shape:", noisy_lines_indices.shape)
        #print("noisy_bands_indices shape:", noisy_bands_indices.shape)

        # Repeat pixels_values to match the dimensions required
        pixels_values = np.tile(pixels_values, (noisy_lines_indices.shape[0], 1, whole_data.shape[1]))

        #print("repeated pixels_values shape:", pixels_values.shape)

        # Check for invalid values
        if np.any(np.isnan(pixels_values)) or np.any(np.isinf(pixels_values)):
            raise ValueError("pixels_values contains NaNs or infinities")

        # Ensure data type compatibility
        if pixels_values.dtype != whole_data.dtype:
            pixels_values = pixels_values.astype(whole_data.dtype)

        #print("pixels_values dtype:", pixels_values.dtype)
        #print("whole_data dtype:", whole_data.dtype)

        # Apply the dropout
        try:
            whole_data[noisy_lines_indices[:, np.newaxis], :, noisy_bands_indices] = pixels_values
        except IndexError as e:
            print(f"IndexError: {e}")
            #print(f"noisy_lines_indices: {noisy_lines_indices}")
            #print(f"noisy_bands_indices: {noisy_bands_indices}")
            #print(f"whole_data.shape: {whole_data.shape}")

        return whole_data








    @staticmethod
    def continuous_column_drop_out(whole_data, noisy_columns_indices, noisy_bands_indices, pixels_values):
        #print("************** applying continuous column dropout ***************************")

        # Ensure pixels_values is not empty and properly initialized
        if pixels_values.size == 0:
            raise ValueError("pixels_values is empty")

        pixels_values = np.asarray(pixels_values)
        #print("pixels_values shape before any operation:", pixels_values.shape)

        # Ensure pixels_values is at least a 3D array for the next steps
        if pixels_values.ndim == 0:
            pixels_values = np.expand_dims(pixels_values, axis=(0, 1, 2))
        elif pixels_values.ndim == 1:
            pixels_values = np.expand_dims(pixels_values, axis=(1, 2))
        elif pixels_values.ndim == 2:
            pixels_values = np.expand_dims(pixels_values, axis=2)

        #print("reshaped pixels_values shape:", pixels_values.shape)

        # Ensure noisy_columns_indices and noisy_bands_indices are arrays
        noisy_columns_indices = np.asarray(noisy_columns_indices)
        noisy_bands_indices = np.asarray(noisy_bands_indices)

        # Check the shapes of indices
       # print("noisy_columns_indices shape:", noisy_columns_indices.shape)
        #print("noisy_bands_indices shape:", noisy_bands_indices.shape)

        # Now we need to repeat pixels_values to match the dimensions required
        pixels_values = np.repeat(pixels_values, whole_data.shape[0], axis=0)  # Repeat for all rows
        pixels_values = np.repeat(pixels_values, len(noisy_columns_indices), axis=1)  # Repeat for all columns
        pixels_values = np.repeat(pixels_values, len(noisy_bands_indices), axis=2)  # Repeat for all bands

        #print("repeated pixels_values shape:", pixels_values.shape)

        # Ensure whole_data is 3D
        if whole_data.ndim != 3:
            raise ValueError("whole_data must be a 3D array")

        # Apply the dropout
        whole_data[:, noisy_columns_indices[:, None], noisy_bands_indices[None, :]] = pixels_values

        return whole_data



    @staticmethod
    def discontinuous_line_drop_out(whole_data, noisy_lines_indices, noisy_columns_indices,
                                noisy_bands_indices, pixels_values):
        mask_add = np.zeros(whole_data.shape)
        mask_mull = np.ones(whole_data.shape)

        for i in range(len(noisy_lines_indices)):
            for j in range(len(noisy_columns_indices)):
                line_idx = noisy_lines_indices[i]
                col_idx = noisy_columns_indices[j]
                band_idx = noisy_bands_indices[0]  # Use the only available band index
                
                mask_add[line_idx, col_idx, band_idx] = pixels_values[i]
                mask_mull[line_idx, col_idx, band_idx] = 0

        return (whole_data * mask_mull) + mask_add

    @staticmethod
    def discontinuous_column_drop_out(whole_data, noisy_columns_indices, noisy_lines_indices,
                                  noisy_bands_indices, pixels_values):
        noisy_lines_indices = np.expand_dims(noisy_lines_indices, axis=1)
        noisy_columns_indices = np.expand_dims(noisy_columns_indices, axis=1)
        pixels_values = np.expand_dims(pixels_values, axis=1)
        pixels_values = np.expand_dims(pixels_values, axis=2)
        pixels_values = np.repeat(pixels_values, len(noisy_lines_indices), axis=1)
        pixels_values = np.repeat(pixels_values, len(noisy_columns_indices), axis=2)

        mask_add = np.zeros(whole_data.shape)
        mask_mull = np.ones(whole_data.shape)
        mask_add[noisy_lines_indices, noisy_columns_indices, noisy_bands_indices] = pixels_values
        mask_mull[noisy_lines_indices, noisy_columns_indices, noisy_bands_indices] = 0

        return (whole_data * mask_mull) + mask_add

    @staticmethod
    def line_stripping(whole_data, noisy_lines_indices, mean_noise, std_noise):
        #print("****** applying line stripping **********")
        #print(noisy_lines_indices)
    
        epsilon = sys.float_info.epsilon

        mean = whole_data[:, noisy_lines_indices, :].mean(axis=2)
        mean = np.repeat(np.expand_dims(mean, axis=2), whole_data.shape[2], axis=2)

        std = whole_data[:, noisy_lines_indices, :].std(axis=2)
        std[np.where(std == 0)[0]] = epsilon
        std = np.repeat(np.expand_dims(std, axis=2), whole_data.shape[2], axis=2)

        whole_data[:, noisy_lines_indices, :] = (std_noise / std) * (
            whole_data[:, noisy_lines_indices, :] - mean + (std / std_noise) * mean_noise)
        return whole_data

    @staticmethod
    def column_stripping(whole_data, noisy_columns_indices, mean_noise, std_noise):
        #print("********************* applying column stripping *********************** ")

        # Print the shape of whole_data and noisy_columns_indices to understand the dimensions
        #print(f"whole_data shape: {whole_data.shape}")
        #print(f"noisy_columns_indices: {noisy_columns_indices}")

        epsilon = sys.float_info.epsilon

        try:
            # Calculate mean along the appropriate axis
            mean = whole_data[:, :, noisy_columns_indices].mean(axis=1)
            #print(f"mean shape: {mean.shape}")
            mean = np.repeat(np.expand_dims(mean, axis=1), whole_data.shape[1], axis=1)

            # Calculate standard deviation along the appropriate axis
            std = whole_data[:, :, noisy_columns_indices].std(axis=1)
           # print(f"std shape: {std.shape}")
            std[np.where(std == 0)[0]] = epsilon
            std = np.repeat(np.expand_dims(std, axis=1), whole_data.shape[1], axis=1)

            # Apply the column stripping transformation
            whole_data[:, :, noisy_columns_indices] = (std_noise / std) * (
                whole_data[:, :, noisy_columns_indices] - mean + (std / std_noise) * mean_noise)
        except IndexError as e:
            print(f"IndexError: {e}")
            #print(f"noisy_columns_indices: {noisy_columns_indices}")
            #print(f"whole_data.shape: {whole_data.shape}")

        return whole_data


    
    @staticmethod
    def region_drop_out(whole_data, xy, width, length, noisy_bands_indices, pixels_values):
        try:
            print("--- Before processing ---")
            print("whole_data shape:", whole_data.shape)
            print("xy:", xy)
            print("width:", width)
            print("length:", length)
            print("noisy_bands_indices:", noisy_bands_indices)
            print("pixels_values shape:", pixels_values.shape if hasattr(pixels_values, 'shape') else pixels_values)
            
            # Ensure the region defined by xy, width, and length fits within whole_data
            if (xy[0] + width) > whole_data.shape[0]:
                width = whole_data.shape[0] - xy[0]
            if (xy[1] + length) > whole_data.shape[1]:
                length = whole_data.shape[1] - xy[1]
            
            # Check if pixels_values is empty or not properly initialized
            if pixels_values.size == 0:
                raise ValueError("pixels_values is empty or improperly initialized.")
            
            # Reshape pixels_values if necessary to match expected dimensions
            if pixels_values.ndim < 3:
                pixels_values = np.expand_dims(pixels_values, axis=(0, 1, 2))  # Assuming pixels_values is (1, 1, 3)
            
            print("--- After resizing ---")
            print("pixels_values shape:", pixels_values.shape)
            
            # Repeat along width, length, and noisy_bands_indices dimensions
            expanded_pixels = np.repeat(pixels_values, width, axis=0)
            expanded_pixels = np.repeat(expanded_pixels, length, axis=1)
            expanded_pixels = np.repeat(expanded_pixels, len(noisy_bands_indices), axis=2)
            
            print("--- After repeating ---")
            print("expanded_pixels shape:", expanded_pixels.shape)
            
            # Apply the modified pixels to the specified region and bands in whole_data
            whole_data[xy[0]:xy[0] + width, xy[1]:xy[1] + length, noisy_bands_indices] = expanded_pixels
            
            print("--- After modification ---")
            print("whole_data shape:", whole_data.shape)
            
            return whole_data
        except Exception as e:
            print("Exception:", e)
            return whole_data

        

    @staticmethod
    def spectral_band_loss(whole_data, noisy_bands_indices):
        try:
            whole_data[:, :, noisy_bands_indices] = np.mean([whole_data[:, :, :-2], whole_data[:, :, 2:]], axis=0)[:, :, noisy_bands_indices - 1]
            return whole_data
        except Exception as e:
            print("spectral band loss")
            
            print(e)
            return whole_data

    @staticmethod
    def salt_and_pepper(whole_data, mask):
        try:
            mask_mul = np.where(mask != 1, 0, mask)
            mask_add = np.where(mask == 1, 0, mask)

            mask_mul = np.repeat(np.expand_dims(mask_mul, axis=2), whole_data.shape[2], axis=2)
            mask_add = np.repeat(np.expand_dims(mask_add, axis=2), whole_data.shape[2], axis=2)

            return (whole_data * mask_mul) + mask_add
        except Exception as e:
            print("salt and pepper")
            
            print(e)
            return whole_data

    @staticmethod
    def spatial_gaussian_noise(whole_data, mask):
        gauss = np.repeat(np.expand_dims(mask, axis=2), whole_data.shape[2], axis=2)

        whole_data = whole_data + gauss
        return whole_data

    @staticmethod
    def spectral_gaussian_noise(whole_data, noisy_bands_indices, noise_values):
        whole_data = whole_data.astype('float64')
        whole_data[:, :, noisy_bands_indices] += noise_values
        return whole_data

    def rotate(self, rotation_angle):
        rotated_image = transform.rotate(self.whole_data, rotation_angle, resize=False, preserve_range=True)
        self.whole_data_copy = rotated_image
        return rotated_image

    def zoom_out(self, zoom_factor):
        if zoom_factor != 0:
            h, w = self.whole_data_copy.shape[:2]
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            zoom_tuple = (zoom_factor,) * 2 + (1,)
            out = np.zeros_like(self.whole_data_copy)

        # Perform zoom operation on the entire image
            for band in range(self.whole_data_copy.shape[2]):
                out[:, :, band] = zoom(self.whole_data_copy[:, :, band], zoom_tuple, order=0)

            self.whole_data_copy = out
            return out
        else:
            return self.whole_data_copy


    def zoom_in(self, zoom_factor):
        if zoom_factor != 0:
            h, w = self.whole_data_copy.shape[:2]
            zh = int(np.ceil(h / zoom_factor))
            zw = int(np.ceil(w / zoom_factor))
            zoom_tuple = (zoom_factor,) * 2 + (1,)
            top = (h - zh) // 2
            left = (w - zw) // 2
            out = np.zeros_like(self.whole_data_copy)

        # Perform zoom operation on the entire image
            for band in range(self.whole_data_copy.shape[2]):
                out[:, :, band] = zoom(self.whole_data_copy[top:top + zh, left:left + zw, band], zoom_tuple, order=0)

            self.whole_data_copy = out
            return out
        else:
            return self.whole_data_copy

def get_activated_element_indices(trf_vector, min_bound, max_bound):
    sub_vector = trf_vector[min_bound: max_bound]
    activated_element_indices = np.where(sub_vector == 1)[0]
    activated_element_indices = activated_element_indices[activated_element_indices % 2 == 1] - 1
    element_indices = sub_vector[activated_element_indices].astype(int)
    return element_indices


def get_activated_spatial_gaussian_noise_indices(trf_vector):
    gaussian_noise_indices = np.arange(2, len(trf_vector), 3)
    activated_noise = np.where(trf_vector[gaussian_noise_indices] != 0)
    noise_indices = gaussian_noise_indices[activated_noise]
    noise_xs = trf_vector[noise_indices - 2].astype(int)
    noise_ys = trf_vector[noise_indices - 1].astype(int)
    noise_values = trf_vector[noise_indices]

    return noise_xs, noise_ys, noise_values


def get_activated_spectral_gaussian_noise_indices(trf_vector):
    #print("*********************** trf vector **********************")
    #print(trf_vector)
    #print("*********************** trf vector **********************")
    
    # Indices in the transformation vector that should contain band indices
    gaussian_noise_indices = np.arange(1, len(trf_vector), 2)
   
    #print("Gaussian noise indices:", gaussian_noise_indices)
    
    # Check which of these indices have non-zero values (activated noise)
    #activated_noise = np.where(trf_vector[gaussian_noise_indices] != 0)
    #print("Activated noise indices:", activated_noise)
    activated_noise = trf_vector[gaussian_noise_indices] != 0
   # print("Activated noise mask:", activated_noise)
    
    # Indices in the transformation vector corresponding to activated noise
    noise_indices = gaussian_noise_indices[activated_noise]
    #print("Noise indices in transformation vector:", noise_indices)
    
    # Extract the band indices and noise values from the transformation vector
    noise_zs = trf_vector[noise_indices - 1].astype(int)
    noise_values = trf_vector[noise_indices]

    #print("Extracted band indices:", noise_zs)
    #print("Extracted noise values:", noise_values)

    return noise_zs, noise_values



def get_salt_pepper_indices(trf_vector, salt_or_pepper):
    noise_indices = np.array(np.where(trf_vector == salt_or_pepper))
    # noise_indices = activated_noise_indices[(activated_noise_indices + 1) % 3 == 0]
    noise_xs_indices = trf_vector[noise_indices - 2].astype(int)
    noise_ys_indices = trf_vector[noise_indices - 1].astype(int)
    return noise_xs_indices, noise_ys_indices


def get_element_from_trf_vector(indices_metadata, transformation_type, index):
    trf_index = indices_metadata[transformation_type + '_index']
    trf_size = indices_metadata[transformation_type + '_size']
    trf_index: trf_size + trf_index
    return trf_index + trf_size + index


def get_gaussian_noise_indices_from_trf_vector(indices_metadata):
    trf_index = indices_metadata['spatial_gaussian_noise_index']
    trf_size = indices_metadata['spatial_gaussian_noise_size']

    spatial_gn_values_indices = np.arange(trf_index + 3, trf_size + trf_index, 3)

    trf_index = indices_metadata['spectral_gaussian_noise_index']
    trf_size = indices_metadata['spectral_gaussian_noise_size']

    spectral_gn_values_indices = np.arange(trf_index + 2, trf_size + trf_index, 2)

    return list(spatial_gn_values_indices) + list(spectral_gn_values_indices)


def format_trf_vector(trf_vector, indices_metadata):
    trf_vector_copy = np.copy(trf_vector)
    mean_index = get_element_from_trf_vector(indices_metadata, "line_col_stripping", -2)
    std_index = get_element_from_trf_vector(indices_metadata, "line_col_stripping", -1)
    angle_index = get_element_from_trf_vector(indices_metadata, "rotation", -1)
    zoom_factor_index = get_element_from_trf_vector(indices_metadata, "zoom", -1)
    gaussian_noise_values_indices = get_gaussian_noise_indices_from_trf_vector(indices_metadata)

    exclude = [mean_index, std_index, angle_index, zoom_factor_index] + gaussian_noise_values_indices

    a = ~np.isin(np.arange(len(trf_vector_copy)), exclude)
    trf_vector_copy[~np.isin(np.arange(len(trf_vector_copy)), exclude)] = np.round(
        trf_vector_copy[~np.isin(np.arange(len(trf_vector_copy)),
                                 exclude)])

    return trf_vector_copy


class VectorDecoder(object):
    

    def __init__(self, whole_data, trf_vector=None):
        params_path_file='F:\\mmsegmentation\\indices_metadata.yaml'
        self.whole_data = whole_data
        self.transformer = Transformer(copy.deepcopy(whole_data))
        self.pixel_values = {"0": np.min(whole_data), "1": np.max(whole_data)}
        self.indices_metadata = self.build_transformation_metadata(params_path_file)
        if trf_vector is not None:
            self.trf_vector = format_trf_vector(trf_vector, self.indices_metadata)


    def set_trf_vector(self, trf_vector):
        self.trf_vector = format_trf_vector(trf_vector, self.indices_metadata)

    @staticmethod
    def build_transformation_metadata(params_file_path):
        with open(params_file_path) as file:
            tr_metadata = yaml.load(file, Loader=yaml.FullLoader)
        return tr_metadata

    

    def apply_continuous_line_col_drop_out(self, line_or_col):
       # print("********************* applying continuous line col dropout *********************** ")

        trf_index = self.indices_metadata['continuous_line_col_drop_out_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['continuous_line_col_drop_out_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            max_lines = self.indices_metadata['max_lines']
            max_bands = self.indices_metadata['max_bands']

            line_indices = get_activated_element_indices(trf_vector, 0, max_lines)
            bands_indices = get_activated_element_indices(trf_vector, max_lines, max_lines + max_bands)
            pixel_key = str(int(trf_vector[-1]))

            # Check if pixel_key exists in self.pixel_values
            if pixel_key not in self.pixel_values:
                raise ValueError(f"Pixel key {pixel_key} not found in pixel_values")

            pixel_values = self.pixel_values[pixel_key]

            # Ensure line_indices and bands_indices are within valid range
            line_indices = np.clip(line_indices, 0, self.whole_data.shape[0] - 1)
            bands_indices = np.clip(bands_indices, 0, self.whole_data.shape[2] - 1)

            #print(f"line_indices: {line_indices}")
           # print(f"bands_indices: {bands_indices}")
           # print(f"pixel_values: {pixel_values}")

            if line_or_col:
                self.whole_data = self.transformer.continuous_line_drop_out(self.whole_data, line_indices, bands_indices, pixel_values)
            else:
                self.whole_data = self.transformer.continuous_column_drop_out(self.whole_data, line_indices, bands_indices, pixel_values)




                    
    def apply_discontinuous_line_col_drop_out(self, line_or_col):
        trf_index = self.indices_metadata['discontinuous_line_col_drop_out_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['discontinuous_line_col_drop_out_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            max_lines = self.indices_metadata['max_lines']
            max_columns = self.indices_metadata['max_columns']
            max_bands = self.indices_metadata['max_bands']

            line_indices = get_activated_element_indices(trf_vector, 0, max_lines)
            columns_indices = get_activated_element_indices(trf_vector, max_lines, max_lines + max_columns)
            bands_indices = get_activated_element_indices(trf_vector, max_lines + max_columns,
                                                        max_lines + max_columns + max_bands)
            pixel_values = self.pixel_values[str(int(trf_vector[-1]))]

            if line_or_col:
                # Repeat line_indices for each column index and tile column_indices for each line index
                line_indices_repeated = np.repeat(line_indices, len(columns_indices))
                columns_indices_tiled = np.tile(columns_indices, len(line_indices))
                
                # Ensure the dimensions match for concatenation
                line_indices_repeated = line_indices_repeated[:len(columns_indices_tiled)]
                columns_indices_tiled = columns_indices_tiled[:len(line_indices_repeated)]
                
                self.whole_data = self.transformer.discontinuous_line_drop_out(self.whole_data, line_indices_repeated, columns_indices_tiled,
                                                                                bands_indices, pixel_values)
        else:
            # Repeat line_indices for each bands index and tile bands_indices for each line index
            line_indices_repeated = np.repeat(line_indices, len(bands_indices))
            bands_indices_tiled = np.tile(bands_indices, len(line_indices))
            
            # Ensure the dimensions match for concatenation
            line_indices_repeated = line_indices_repeated[:len(bands_indices_tiled)]
            bands_indices_tiled = bands_indices_tiled[:len(line_indices_repeated)]
            
            self.whole_data = self.transformer.discontinuous_column_drop_out(self.whole_data, line_indices_repeated,
                                                                            columns_indices, bands_indices_tiled,
                                                                            pixel_values)


    def apply_line_stripping(self, line_or_col):
        #print("********************* applying line stripping *********************** ")
        trf_index = self.indices_metadata['line_col_stripping_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['line_col_stripping_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            max_lines = self.indices_metadata['max_lines']

            line_indices = get_activated_element_indices(trf_vector, 0, max_lines)
            mean = trf_vector[-2]
            std = trf_vector[-1]

            if line_or_col:
                self.whole_data = self.transformer.line_stripping(self.whole_data, line_indices, mean, std)
                #print("********************* exiting line stripping *********************** ")
                
            else:
                self.whole_data = self.transformer.column_stripping(self.whole_data, line_indices, mean, std)
                #print("********************* exiting line stripping *********************** ")
                
    def apply_line_col_transformations(self):
        line_or_col = self.trf_vector[self.indices_metadata['line_col_transformation_index']]
        #print("After line_or_col assignment:", line_or_col)

        self.apply_continuous_line_col_drop_out(line_or_col)
        #self.apply_discontinuous_line_col_drop_out(line_or_col)
        self.apply_line_stripping(line_or_col)

    def apply_region_drop_out(self):
        trf_index = self.indices_metadata['region_drop_out_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['region_drop_out_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            max_bands = self.indices_metadata['max_bands']

            bands_indices = get_activated_element_indices(trf_vector, 0, max_bands)
            x, y, width, length = trf_vector[-5:-1].astype(int)
            pixel_values = self.pixel_values[str(int(trf_vector[-1]))]

            self.whole_data = self.transformer.region_drop_out(self.whole_data, (x, y), width, length, bands_indices,
                                                               pixel_values)

    def apply_spectral_band_loss(self):
        trf_index = self.indices_metadata['spectral_band_loss_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['spectral_band_loss_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            max_bands = self.indices_metadata['max_bands_for_sbl']

            bands_indices = get_activated_element_indices(trf_vector, 0, max_bands)

            self.whole_data = self.transformer.spectral_band_loss(self.whole_data, bands_indices)

    def apply_salt_and_pepper(self):
        trf_index = self.indices_metadata['salt_pepper_noise_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['salt_pepper_noise_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            salt_mask = np.zeros_like(self.whole_data[:, :, 0])
            pepper_mask = np.zeros_like(self.whole_data[:, :, 0])

            salt_xs, salt_ys = get_salt_pepper_indices(trf_vector, -1)
            pepper_xs, pepper_ys = get_salt_pepper_indices(trf_vector, -2)

            salt_mask[salt_xs, salt_ys] = 1
            pepper_mask[pepper_xs, pepper_ys] = 1

            for i in range(self.whole_data.shape[-1]):
                min_val = self.whole_data[:, :, i].min()
                max_val = self.whole_data[:, :, i].max()

                self.whole_data[:, :, i][salt_mask == 1] = max_val
                self.whole_data[:, :, i][pepper_mask == 1] = min_val
                
                """
                if self.pixel_values['0'] != 0:
                    self.whole_data[:, :, i][salt_mask == 1] = self.pixel_values['0']
                if self.pixel_values['1'] != 0:
                    self.whole_data[:, :, i][pepper_mask == 1] = self.pixel_values['1']
                    """

    def apply_spatial_gaussian_noise(self):
        trf_index = self.indices_metadata['spatial_gaussian_noise_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['spatial_gaussian_noise_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            mask = np.zeros(self.whole_data.shape[:2])
            gn_xs, gn_ys, noise_values = get_activated_spatial_gaussian_noise_indices(trf_vector)

            # Clip indices to ensure they're within the valid range
            gn_xs = np.clip(gn_xs, 0, self.whole_data.shape[0] - 1)
            gn_ys = np.clip(gn_ys, 0, self.whole_data.shape[1] - 1)

            # Set noise values in the mask array
            mask[gn_xs, gn_ys] = noise_values

            # Create a gaussian noise array with the same shape as whole_data
            gauss = np.repeat(mask[:, :, np.newaxis], self.whole_data.shape[2], axis=2)

            self.whole_data = self.whole_data + gauss

    def apply_spectral_gaussian_noise(self):
        trf_index = self.indices_metadata['spectral_gaussian_noise_index']
       # print("************ trf indexxx *****************")
       # print(trf_index)
       # print("************ trf indexxx *****************")
        
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['spectral_gaussian_noise_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]
         #   print("****************** trf vector *************************************")
           # print(trf_vector)
           # print("****************** trf vector *************************************")
            
            bands_indices, noise_values = get_activated_spectral_gaussian_noise_indices(trf_vector)
           # print("****************** band indices *************************************")
           # print(bands_indices)
           # print("****************** band indices *************************************")

            self.whole_data = self.transformer.spectral_gaussian_noise(self.whole_data, bands_indices, noise_values)


    def apply_rotation(self):
        trf_index = self.indices_metadata['rotation_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['rotation_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            rotation_angle = trf_vector[2]
            self.transformer.rotate(rotation_angle)  # Remove self.whole_data argument


    def apply_zoom(self):
        trf_index = self.indices_metadata['zoom_index']
        activation = self.trf_vector[trf_index]
        if activation:
            trf_size = self.indices_metadata['zoom_size']
            trf_vector = self.trf_vector[trf_index + 1: trf_size + trf_index]

            zoom_factor = trf_vector[2]
            if zoom_factor > 1:
                self.whole_data = self.transformer.zoom_in(self.whole_data, zoom_factor)
            else:
                self.whole_data = self.transformer.zoom_out(zoom_factor)

    def apply_rotations_and_zoom(self):
        self.apply_rotation()
        self.apply_zoom()

    def apply_rotations_and_zoom_vect(self, i):
        self.apply_rotation(i)
        self.apply_zoom(i)

    def apply_rotations_and_zooms_final(self):
        self.apply_rotations_and_zoom()
    def apply_all_transformations(self):
        self.apply_line_col_transformations()
        self.apply_region_drop_out()#done
        self.apply_spectral_band_loss()#done
        self.apply_salt_and_pepper()#done
        self.apply_spatial_gaussian_noise()#done
        self.apply_spectral_gaussian_noise()#done
        #self.apply_rotations_and_zooms_final()

hsi_data_trf=image_array
whole_data = hsi_data_trf  # Example image data
print(shape)


def psnr_calc(originals, transformeds):
    #print("psnr is being calculated")
    epsilon = sys.float_info.epsilon
    # Ensure pixel values are in the range [0, max_val]
    
    
    mse = np.mean(np.square(originals - transformeds))
    
    psnr_value = 20 * np.log10(np.max(originals)) - 10 * np.log10(mse+epsilon )
    
    return psnr_value

def calculate_iou_whole_image(pred_clean, pred_noisy, num_classes):
    # Ensure the arrays have the same shape
    #assert pred_clean.shape == pred_noisy.shape, "Shape mismatch between clean and noisy predictions"
    
    intersection = 0
    union = 0
    
    for cls in range(num_classes):
        pred_clean_class = (pred_clean == cls)
        pred_noisy_class = (pred_noisy == cls)
        
        # Calculate intersection and union for each class
        cls_intersection = np.sum(pred_clean_class & pred_noisy_class)
        cls_union = np.sum(pred_clean_class | pred_noisy_class)
        
        intersection += cls_intersection
        union += cls_union
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    # Calculate IoU
    iou = intersection / union
    return iou




    
    
#def fitness_function(pred_mask,gt_mask,pred_box, gt_box):
def max_psnr(original_data):
    return psnr_calc(original_data,original_data)

def fitness_function(pred_mask, gt_mask, psnr, confidence_threshold=0.5):
    # Check if the predicted mask is void (below confidence threshold)
    if np.max(pred_mask) < confidence_threshold:
        fitness_score = 0  # Assign a low fitness score for void predictions
    else:
        # Convert gt_mask to match the model's expected classes
       
        
        iou = calculate_iou_whole_image(pred_mask, gt_mask,19)
        
        print("IoU: ", iou)
        
        # Plot matching and unmatched areas
        
        max_psnrr = max_psnr(hsi_data_trf)
        fitness = psnr - iou * 100

    return fitness, iou, psnr

def main_fitness_function(ga_instance, solution, solution_idx):
    global average_fitness
    psnr_positive = 0
    fitness_score = 0
    average = []
    global iteration
    
    print("********************* New Iteration *****************************")
    print("****************************************************************")
    print(hsi_data_trf)
    print("****************************************************************")
    
    max_min_generator = MaxMinGenerator(shape)
    encoder = VectorEncoder(shape)
    decoder = VectorDecoder(hsi_data_trf)
    decoder.set_trf_vector(solution)

    # Always start with the original clean image
    original_image = image_copy.copy()
    decoder.whole_data = original_image
    
    decoder.apply_all_transformations()

    transformed_image_data = decoder.whole_data
    print("************************ Transformation Vector ***************************")
    print(solution)
    print("************************ Transformation Vector ***************************")
    print("****************************************************************")
    
    print(transformed_image_data)
    print("****************************************************************")
    
    psnr_value = psnr_calc(image_copy, transformed_image_data)
    print(psnr_value)
    
    inferencer = MMSegInferencer(model='deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024', device='cuda')
    result = inferencer(transformed_image_data, out_dir='F:\\mmsegmentation\\predited_masks', pred_out_dir="pred")
    print(result["predictions"])
    noised_mask = result["predictions"]
    
    root_directory = "F:\\mmsegmentation\\cityscapes\\gtFine\\val\\frankfurt\\"

    # List all files in the directory
    files = os.listdir(root_directory)
    clean_mask_array = Image.open("F:\\mmsegmentation\\cityscapes\\leftImg8bit\\val\\frankfurt\\frankfurt_000001_083852_leftImg8bit.png")

    # Assuming there are images in the directory, take the first one
    first_image_path = os.path.join(root_directory, files[3])
    print(files[3])

    # Load the image
    image = Image.open(first_image_path)

    # Convert the PIL Image to a NumPy array
    image_array = np.array(image)
    
    fitness, iou, p = fitness_function(noised_mask, np.array(clean_mask_array), psnr_value)

    # Normalize PSNR and IoU
    normalized_psnr = psnr_value / 100.0  # Assuming max PSNR is 100 for normalization
    normalized_iou = iou  # Assuming IoU is already in [0, 1] range

    # Combine into a fitness score
    fitness_score = normalized_psnr - normalized_iou  # Maximize PSNR, minimize IoU

    # Penalize solutions with PSNR below the threshold
    psnr_threshold = 0
    if psnr_value < psnr_threshold:
        penalty = (psnr_threshold - psnr_value) ** 2  # Example penalty function
        fitness_score -= penalty
    
    # Save the noisy image only if PSNR is above the threshold
    if psnr_value >= psnr_threshold:
        tifffile.imwrite("F:\\mmsegmentation\\adv_examples_citytscapes\\inputt--"+str(iou)+"---"+str(psnr_value)+".tif", transformed_image_data, photometric="RGB")
        print("Image saved as noisy_image.tif")
    else:
        print("PSNR below threshold, image not saved.")
    
    return fitness_score



def psnr_calc(original_image, transformed_image):
    mse = np.mean((original_image - transformed_image) ** 2)
    if mse == 0:
        return 100  # Arbitrary high value for no distortion
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def fitness_function(pred_mask, gt_mask, psnr, confidence_threshold=0.5):
    # Check if the predicted mask is void (below confidence threshold)
    if np.max(pred_mask) < confidence_threshold:
        fitness_score = 0  # Assign a low fitness score for void predictions
    else:
        # Convert gt_mask to match the model's expected classes
       
        
        iou = calculate_iou_whole_image(pred_mask, gt_mask,19)
        
        print("IoU: ", iou)
        
        # Plot matching and unmatched areas
        
        max_psnrr = max_psnr(hsi_data_trf)
        fitness = psnr - iou * 100

    return fitness, iou, psnr

def wrapped_fitness_function(image, mask):
    def main_fitness_function(ga_instance, solution, solution_idx):
        gc.collect()

        max_min_generator = MaxMinGenerator(shape)
        encoder = VectorEncoder(shape)
        decoder = VectorDecoder(hsi_data_trf)
        decoder.set_trf_vector(solution)

        original_image = image.copy()
        decoder.whole_data = original_image

        decoder.apply_all_transformations()

        transformed_image_data = decoder.whole_data

        psnr_value = psnr_calc(image, transformed_image_data)

        inferencer = MMSegInferencer(model='deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024', device='cuda')
        result = inferencer(transformed_image_data, out_dir='F:\\mmsegmentation\\predited_masks', pred_out_dir="pred")
        noised_mask = result["predictions"]

        _, iou, _ = fitness_function(noised_mask, mask, psnr_value)

        # Save images and transformation vectors with PSNR < 0.1
        if psnr_value < 0.1:
            save_low_psnr_image_and_vector(transformed_image_data, solution, psnr_value, 'F:\\mmsegmentation\\low')

        # Ensure PSNR is above the threshold
        psnr_threshold = 15
        if psnr_value < psnr_threshold:
            return 1 / 999999999999999  # Heavily penalize low PSNR values

        # Minimize the fitness function by returning the inverse of IoU
        minimized_fitness =  iou * 100  # Adding a small value to avoid division by zero

        return 1 / (minimized_fitness + 1e-6)

    return main_fitness_function

clean_mask_array = Image.open("F:\\mmsegmentation\\cityscapes\\gtFine\\val\\frankfurt\\frankfurt_000001_083852_gtFine_labelTrainIds.png")
print(np.array(clean_mask_array))

def generate_random_transformation_vectors(size):
    initial_pop = []

    for _ in range(size):
        print(size)
        shape=hsi_data_trf.shape
        params_path = "F:\\mmsegmentation\\parameters_metadata.yaml"

        template_path = "F:\\mmsegmentation\\template.yaml"
        output_path = "F:\\mmsegmentation\\indices_metadata.yaml"
        with open(params_path) as file:
            params = yaml.safe_load(file)
            

        output = generate_indices_metadata(params, template_path, shape)  # Unpack the shape tuple
        print("1")

        max_min_generator = MaxMinGenerator(shape)
        print("2")
        
        encoder = VectorEncoder(shape)
        print("3")
        
        transformation_vector = encoder.construct_random_tr_vector()
        print("4")
        
        initial_pop.append(transformation_vector)
        print("5")
        gc.collect()

        
    return initial_pop

# Generate the initial population of transformation vectors
initial_population = generate_random_transformation_vectors(20)

def run_ga_on_directory(image_dir, mask_dir, output_dir):
    torch.cuda.empty_cache()
    gc.collect()

    for filename in tqdm(os.listdir(image_dir), desc="Processing Images"):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_file_path = os.path.join(image_dir, filename)
            mask_filename = filename.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
            mask_file_path = os.path.join(mask_dir, mask_filename)

            if os.path.exists(mask_file_path):
                image = np.array(Image.open(image_file_path))
                mask = np.array(Image.open(mask_file_path))

                # Run the GA for this image-mask pair
                best_solution, final_image = genetic_algorithm(image, mask)

                # Calculate PSNR and IoU
                image_psnr = calculate_psnr(image, final_image)
                inferencer = MMSegInferencer(model='deeplabv3_r50-d8_4xb2-40k_cityscapes-512x1024', device='cuda')
                result_corrupted = inferencer(np.array(final_image), out_dir='F:\\mmsegmentation\\predited_masks', pred_out_dir="pred")

                image_iou = calculate_iou_whole_image(mask, result_corrupted["predictions"], 19)

                # Save the best transformation vector, final image, PSNR, and IoU
                save_result(output_dir, filename, best_solution, final_image, image_psnr, image_iou)
from tqdm import tqdm
def save_low_psnr_image_and_vector(image, transformation_vector, psnr_value, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_filename = f'low_psnr_{psnr_value:.4f}.png'
    vector_filename = f'low_psnr_{psnr_value:.4f}_vector.txt'

    image_path = os.path.join(output_dir, image_filename)
    vector_path = os.path.join(output_dir, vector_filename)

    Image.fromarray(image).save(image_path)

    with open(vector_path, 'w') as file:
        file.write(str(transformation_vector))
        
        
def genetic_algorithm(image, mask):
    torch.cuda.empty_cache()
    gc.collect()

    # Define the additional data to pass to the fitness function
    additional_data = {'image': image, 'mask': mask}

    # GA configuration
    ga_instance = pygad.GA(num_generations=50,
                           num_parents_mating=2,
                           fitness_func=wrapped_fitness_function(image, mask),
                           sol_per_pop=4,
                           num_genes=len(initial_population[0]), # Adjust based on the length of the transformation vector
                           initial_population=initial_population,
                           mutation_type="random",
                           mutation_percent_genes=10,
                           mutation_by_replacement=True,
                           random_mutation_min_val=0.0,
                           random_mutation_max_val=1.0,
                           crossover_probability=0.7,
                           parent_selection_type="tournament",
                           stop_criteria=["saturate_10"],
                          )

    # Run the GA
    ga_instance.run()

    # Get the best solution and apply the transformations to get the final image
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    decoder = VectorDecoder(hsi_data_trf)
    decoder.set_trf_vector(best_solution)
    decoder.whole_data = image.copy()
    decoder.apply_all_transformations()
    final_image = decoder.whole_data

    return best_solution, final_image

def calculate_psnr(original_image, final_image):
    return psnr_calc(original_image, final_image)



def calculate_iou_whole_image(pred_clean, pred_noisy, num_classes):
    # Ensure the arrays have the same shape
    #assert pred_clean.shape == pred_noisy.shape, "Shape mismatch between clean and noisy predictions"
    
    intersection = 0
    union = 0
    
    for cls in range(num_classes):
        pred_clean_class = (pred_clean == cls)
        pred_noisy_class = (pred_noisy == cls)
        
        # Calculate intersection and union for each class
        cls_intersection = np.sum(pred_clean_class & pred_noisy_class)
        cls_union = np.sum(pred_clean_class | pred_noisy_class)
        
        intersection += cls_intersection
        union += cls_union
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    # Calculate IoU
    iou = intersection / union
    return iou

def save_result(output_dir, filename, best_solution, final_image, image_psnr, image_iou):
    result_file = os.path.join(output_dir, f"result_{filename}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Best Solution: {str(best_solution)}\n")
        f.write(f"PSNR: {image_psnr}\n")
        f.write(f"IoU: {image_iou}\n")

    # Save the final image
    tifffile.imwrite(os.path.join(output_dir, f"final_image_{filename}.tif"), final_image, photometric="RGB")

# Example usage
image_directory = 'F:\\mmsegmentation\\cityscapes\\leftImg8bit\\val\\frankfurt'
mask_directory = 'F:\\mmsegmentation\\cityscapes\\gtFine\\val\\frankfurt'
output_directory = "F:\\mmsegmentation\\output2\\frankfurt"

while True:
    run_ga_on_directory(image_directory, mask_directory, output_directory)







