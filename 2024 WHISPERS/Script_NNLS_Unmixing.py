import numpy as np 
import pandas as pd 
import spectral.io.envi as envi
from scipy.optimize import nnls
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from collections import defaultdict
import statistics
import time


if __name__ == '__main__':
    print("Running NNLS_Unmixing.py. Run from 'Run' notebook")
else:
    
    class NNLS:
        def __init__(self, image_hdr_filename, image_filename, pixel_location, spectral_library_filename):
            self.load_new_image(image_hdr_filename, image_filename)
            self.load_pixel_location_info(pixel_location)
            self.load_new_spectral_library(spectral_library_filename)
            self.load_chemical_data()
            self.reshape_image()                        

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
        
        def load_pixel_location_info(self, pixel_location):
            if pixel_location.endswith('.csv'):
                self.df = pd.read_csv(pixel_location)
            else:
                print("WARNING: pixel_location should be a .csv file! ")

        def load_chemical_data(self):
            self.chemicaldf = pd.read_csv("mineraldata2.csv", encoding='latin1')              
        
        def generate_pixel_samples(self, mineral_type=None, region=None):
            self.ROI = []
            samples = []

            if mineral_type == 1:
                self.target_mineral = "alunite"
                if region == 1:
                    self.ROI = ["alunite hill 1"]
                else:
                    self.ROI = ["alunite hill 2"]    
            else:
                self.target_mineral = "kaolinite"
                if region == 1:
                    self.ROI = ["kaolinite region 1"]
                else:
                    self.ROI = ["kaolinite region 2"]

            samples = self.df[self.df['Name'].isin(self.ROI)].iloc[:, [2, 3]].values.tolist()
            print(self.ROI)
            return samples      
              
        def selectedindex_fit(self, mineral_type=None, region=None):            
            self.technique = "NNLS Regression"
            pixel_samples = self.generate_pixel_samples(mineral_type=mineral_type, region=region)
            
            self.mineral_data = defaultdict(list)
            self.pixel_y_data = {}
            self.rmse_list = []
            self.computation_time = []
            self.model_size = []
            inclusion_count = 0

            if self.target_mineral.lower() == 'alunite':
                keywords = ['alunite', 'alun']
            elif self.target_mineral.lower() == 'kaolinite':
                keywords = ['kaolin', 'kaolinite', 'kaolin/smect', 'kaosmec']
            else:
                keywords = [self.target_mineral.lower()]

            for pixel_sample in pixel_samples: 
                start_time = time.time()
                
                y_index = tuple(pixel_sample)
                self.y = self.df[(self.df.iloc[:, 2] == y_index[0]) & (self.df.iloc[:, 3] == y_index[1])].iloc[:, 4:].values.flatten()
                self.pixel_y_data[y_index] = self.y

                self.X = self.spectral_library.T
                self.model_coefficients, _ = nnls(self.X, self.y)

                self.non_zero_indices = np.where(self.model_coefficients != 0)[0]
                self.non_zero_coefficients = np.round(self.model_coefficients[self.non_zero_indices],2)
                self.non_zero_spectral_names = [self.spectra_names[index] for index in self.non_zero_indices]
                
                y_infer = np.dot(self.X[:, self.model_coefficients != 0], self.non_zero_coefficients)
                pixel_rmse = np.sqrt(mean_squared_error(self.y, y_infer))

                end_time = time.time()
                elapsed_time = end_time - start_time                
                
                self.mineral_data[f'{pixel_sample}'] = list(zip(self.non_zero_spectral_names, self.non_zero_coefficients))
                self.rmse_list.append(pixel_rmse)
                self.computation_time.append(elapsed_time)
                self.model_size.append(len(self.non_zero_coefficients))

                if any(keyword in name.lower() for keyword in keywords for name in self.non_zero_spectral_names):
                    inclusion_count += 1            
            
            self.rmse_mean = np.mean(self.rmse_list)
            self.rmse_std = np.std(self.rmse_list)

            self.computation_time_mean = np.mean(self.computation_time)
            self.model_size_mean = np.mean(self.model_size)      

            total_pixel_samples = len(pixel_samples)
            self.target_mineral_proportion = inclusion_count / total_pixel_samples if total_pixel_samples > 0 else 0  
        
            print(f"Number of pixel samples analyzed: {total_pixel_samples}")
            print(f"Number of models including the target mineral: {inclusion_count}")
            print(f"Proportion of models including the target mineral: {self.target_mineral_proportion:.4f}")

            
            self.plot_median_rmse_spectrum()
            self.plot_metrics_distributions()

        def plot_median_rmse_spectrum(self):
            n = len(self.rmse_list)
            if n % 2 == 0:
                sorted_rmse_list = sorted(self.rmse_list)
                median_rmse = sorted_rmse_list[n // 2 - 1]
            else:
                median_rmse = statistics.median(self.rmse_list)
            median_index = self.rmse_list.index(median_rmse)

            
            median_pixel_sample = list(self.mineral_data.keys())[median_index]


            non_zero_data = self.mineral_data[median_pixel_sample]
            non_zero_indices = [self.spectra_names.index(name) for name, _ in non_zero_data]
            non_zero_coefficients = [value for _, value in non_zero_data]

            
            if len(non_zero_indices) != len(non_zero_coefficients):
                raise ValueError(f"Mismatch in dimensions: {len(non_zero_indices)} indices vs {len(non_zero_coefficients)} coefficients")

            
            median_inferred_spectrum = np.dot(self.X[:, non_zero_indices], non_zero_coefficients)

            plt.figure(figsize=(13, 8))
            for pixel_sample in self.pixel_y_data:
                observed_spectrum = self.pixel_y_data[pixel_sample]                
                plt.plot(observed_spectrum)
                #plt.plot(observed_spectrum, color='gray', alpha=0.5, label='Observed Spectrum' if pixel_sample == list(self.pixel_y_data.keys())[0] else "")

            # Plot the median inferred spectrum
            if median_inferred_spectrum is not None:
                plt.plot(median_inferred_spectrum, linestyle='--', linewidth=2, c='black', label='Median Inferred Spectrum')

            plt.suptitle(self.technique)
            plt.title(f"Observed Spectra vs. Inferred Spectrum (Median RMSE: {median_rmse:.4f})")
            plt.xlabel('Wavelength Index')
            plt.ylabel('Intensity')
            plt.legend()

        def plot_metrics_distributions(self, ):
             # Plot the distribution of RMSE: Average
            plt.figure(figsize=(10, 6))
            plt.suptitle(self.technique)

            # RMSE Distribution
            plt.hist(self.rmse_list, bins=20, color='skyblue', edgecolor='black')
            plt.title('Distribution of RMSE')
            plt.xlabel('RMSE')
            plt.ylabel('Frequency')
            # Add mean and variance as text annotations
            plt.text(0.95, 0.85, f"Mean: {self.rmse_mean:.4f}\nStandard Deviation: {self.rmse_std:.4f}", 
                    transform=plt.gca().transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))            

            plt.tight_layout()
            plt.show()        