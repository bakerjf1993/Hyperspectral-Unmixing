import numpy as np 
import pandas as pd 
import spectral.io.envi as envi
from itertools import combinations
from statsmodels.regression.linear_model import OLS
from itertools import chain
from scipy.optimize import nnls
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import matplotlib.image as img
from sklearn.metrics import mean_squared_error
import math
from collections import defaultdict
from collections import Counter
import statistics


if __name__ == '__main__':
    print("Running NNLS_Unmixing.py. Run from 'Run' notebook")
else:
    
    class NNLS:
        def __init__(self, image_hdr_filename, image_filename, spectral_library_filename):
            self.load_new_image(image_hdr_filename, image_filename)
            self.load_new_spectral_library(spectral_library_filename)
            self.reshape_image()
            f, axarr = plt.subplots(1,2,figsize=(15, 15))
            axarr[0].imshow(self.show_image())
            axarr[1].imshow(self.show_paper_im())                        

        def load_new_image(self, image_hdr_filename, image_filename):
            if image_hdr_filename.endswith('.hdr'):
                self.image = envi.open(image_hdr_filename, image_filename)
                self.image_arr = self.image.load()  
                self.n_rows, self.n_cols, self.n_imbands = self.image_arr.shape
                self.abundance_maps = None
            else:
                print("WARNING: library should be a .hdr file!")

        def load_new_spectral_library(self, spectral_library_filename):
            if spectral_library_filename.endswith('.hdr'):
                lib = envi.open(spectral_library_filename)
                self.wavelengths = lib.bands.centers
                self.n_bands = len(self.wavelengths)
                self.n_spectra= len(lib.names)
                self.spectral_library = lib.spectra
                self.spectra_names = lib.names  
            else:
                print("WARNING: library should be a .hdr file!")
        
        def reshape_image(self):
            reshaped_image = self.image_arr.reshape(self.n_rows * self.n_cols, self.n_bands)
            return reshaped_image
        
        def generate_pixel_samples(self, mineral_type=None, pixel_radius=None):           
            n_rows, n_cols, _ = self.image_arr.shape
            samples = []
            start_points = []
            if mineral_type == 1:
                start_points.extend([(305, 90)])# (100, 110)])
                adjacent_pixels = 1
                if pixel_radius == None:
                    pixel_radius = np.random.randint(1, 20)
            elif mineral_type == 2:
                start_points.extend([(290, 48)])# (310, 55)])  # montmorillonite
                adjacent_pixels = 1
                if pixel_radius == None:
                    pixel_radius = np.random.randint(1, 20)
            else:
                start_points.extend([(290, 290)])# (25, 150)])  # kaolinite
                adjacent_pixels = 1
                if pixel_radius == None:
                    pixel_radius = np.random.randint(1, 20)

            print("Sample Centers")
            for start_row, start_col in start_points:
                # Randomly select a pixel within the region
                random_offset_row = np.random.randint(-adjacent_pixels, adjacent_pixels + 1)
                random_offset_col = np.random.randint(-adjacent_pixels, adjacent_pixels + 1)
                random_row = max(0, min(n_rows, start_row + random_offset_row))
                random_col = max(0, min(n_cols, start_col + random_offset_col))
                print(random_row,random_col)

                
                # Generate a grid of samples around the randomly selected pixel
                for i in range(-pixel_radius, pixel_radius + 1):
                    for j in range(-pixel_radius, pixel_radius + 1):
                        row = max(0, min(n_rows - 1, random_row + i))
                        col = max(0, min(n_cols - 1, random_col + j))
                        samples.append((row, col))
            return samples
        
              
        def selectedindex_fit(self, mineral_type=None, pixel_samples=None, pixel_radius=None):
            if pixel_samples is None:
                pixel_samples = self.generate_pixel_samples(mineral_type=mineral_type, pixel_radius=pixel_radius)
                print(f'Number of samples = (1+2r)^2 :{len(pixel_samples)}')                
                print(f"Samples: {pixel_samples}")
            
            self.mineral_data = defaultdict(list)
            
            for pixel_sample in pixel_samples:
                # Fit the spectral library to each pixel sample
                y_index = pixel_sample

                self.technique = "NNLS Regression"
                self.y = self.image_arr[y_index[0], y_index[1], :].flatten()

                self.X = self.spectral_library.T

                self.model_coefficients, _ = nnls(self.X, self.y)

                self.non_zero_indices = np.where(self.model_coefficients != 0)[0]
                self.non_zero_coefficients = np.round(self.model_coefficients[self.non_zero_indices],2)
                self.non_zero_spectral_names = [self.spectra_names[index] for index in self.non_zero_indices]

                y_infer = np.dot(self.X[:, self.model_coefficients != 0], self.non_zero_coefficients)
                self.rmse = np.sqrt(mean_squared_error(self.y, y_infer))

                
                self.mineral_data[f'{pixel_sample}'] = list(zip(self.non_zero_spectral_names, self.non_zero_coefficients))

            self.final_summary()
            
            return self

        def final_summary(self, top_n=None):
            if top_n is None:
                # Determine the number of top minerals based on the most common minerals in the model
                counts = [len(abundances) for abundances in self.mineral_data.values()]
                top_n = statistics.mode(counts)

            # Count the occurrences of each mineral
            mineral_counts = {}
            for abundances in self.mineral_data.values():
                for mineral_abundance in abundances:
                    mineral_name = mineral_abundance[0]
                    if mineral_name not in mineral_counts:
                        mineral_counts[mineral_name] = 0
                    mineral_counts[mineral_name] += 1

            # Get the top n most common minerals
            top_minerals = sorted(mineral_counts, key=mineral_counts.get, reverse=True)[:top_n]

            mineral_abundances = {mineral: [] for mineral in top_minerals}

            # Populate the dictionary with abundances for each mineral
            for abundances in self.mineral_data.values():
                for mineral_abundance in abundances:
                    mineral_name, abundance = mineral_abundance
                    if mineral_name in mineral_abundances:
                        mineral_abundances[mineral_name].append(abundance)

            # Initialize a dictionary to store the most common abundance for each top mineral
            most_common_abundances = {}

            # Iterate through the top minerals to find their most common abundance
            for mineral, abundances in mineral_abundances.items():
                most_common_abundance = max(set(abundances), key=abundances.count)
                most_common_abundances[mineral] = most_common_abundance

            # Print the top n most common minerals and their associated most common abundances
            print(f"Top {top_n} most common minerals and their associated most common abundances:")
            for mineral, abundance in most_common_abundances.items():
                print(f"{mineral}: Most Common Abundance = {abundance:.2f}")

            return self

        def show_paper_im(self):
            im = img.imread('cuperite_paper.png') 
            return im

        def show_image(self,pix=[]):
            
            plt.title("File_Image")
            skip = int(self.n_bands / 4)
            imRGB = np.zeros((self.n_rows,self.n_cols,3))
            for i in range(3):
                imRGB[:,:,i] = self.stretch(self.image_arr[:,:,i * skip])
            for loc in pix:
                imRGB[loc[0],loc[1],0] = 1
                imRGB[loc[0],loc[1],1] = 0
                imRGB[loc[0],loc[1],2] = 0
            return imRGB

        def show_pca(self):
            n_components = 30
            pca = PCA(n_components=n_components)
            pca.fit(self.image_arr2d.T)
            self.imag_pca = pca.transform(self.image_arr2d.T)
            self.ImPCA = np.reshape(self.imag_pca, (self.n_rows,self.n_cols,n_components))
            imRGBpca1 = np.zeros((self.n_rows,self.n_cols,3))
            for i in range(3):
                imRGBpca1[:,:,i] = self.stretch(self.ImPCA[:,:,i])        
            imRGBpca2 = np.zeros((self.n_rows,self.n_cols,3))
            for i in range(3):
                imRGBpca2[:,:,i] = self.stretch(self.ImPCA[:,:,i+3])

        def stretch(self, arr):
            low = np.percentile(arr, 1)
            high = np.percentile(arr, 99)
            arr[arr<low] = low
            arr[arr>high] = high
            return np.clip(np.squeeze((arr-low)/(high-low)), 0, 1)

        def stretch_05(self, arr):
            low = np.percentile(arr, 0.5)
            high = np.percentile(arr, 99.5)
            arr[arr<low] = low
            arr[arr>high] = high
            return np.clip(np.squeeze((arr-low)/(high-low)), 0, 1)
            
            

             

            
            


            
