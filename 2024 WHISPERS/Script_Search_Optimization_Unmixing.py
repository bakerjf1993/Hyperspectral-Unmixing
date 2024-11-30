import numpy as np 
import pandas as pd 
import spectral.io.envi as envi
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from itertools import chain
from scipy.stats import t
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
import warnings
from collections import defaultdict
from collections import Counter
import statistics
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.image as img
from sklearn.decomposition import PCA
import time

if __name__ == '__main__':
    print("Running Search_Optimization_Unmixing.py. Run from 'Run' notebook")
else:
    warnings.simplefilter(action='ignore', category=FutureWarning)

    class Search_Optimization:
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
            
            # Use the isin method for the comparison
            samples = self.df[self.df['Name'].isin(self.ROI)].iloc[:, [2, 3]].values.tolist()
            print(self.ROI)
            return samples

        def depth_first_search(self, y_index, threshold_in):
            self.y = self.df[(self.df.iloc[:, 2] == y_index[0]) & (self.df.iloc[:, 3] == y_index[1])].iloc[:, 4:].values.flatten()

            self.X = self.spectral_library.T

            initial_list = []
            included = list(initial_list)
            while True:
                changed=False
                excluded = [col for col in range(self.X.shape[1]) if col not in included]
                new_pval = pd.Series(index=excluded)

                for new_column in excluded:
                    X_subset = np.column_stack([self.X[:, included], self.X[:, new_column]])  # Add new column
                    model = sm.OLS(self.y, X_subset).fit()
                    
                    p_value = model.pvalues[-1]
                    new_pval[new_column] = p_value

                best_pval = new_pval.min()
                if best_pval < threshold_in:
                    best_feature = new_pval.idxmin()
                    included.append(best_feature)
                    changed=True

                if not changed:
                    break
            return included
        
        def breadth_first_search(self, y_index, threshold_out):
            self.y = self.df[(self.df.iloc[:, 2] == y_index[0]) & (self.df.iloc[:, 3] == y_index[1])].iloc[:, 4:].values.flatten()

            self.X = self.spectral_library.T

            included = list(range(self.X.shape[1]))
            while True:
                changed=False
                
                model = sm.OLS(self.y, self.X[:, included]).fit()
                
                pvalues = model.pvalues
                worst_feature_index = np.argmax(pvalues)
                worst_feature = included[worst_feature_index]
                
                if pvalues[worst_feature_index] > threshold_out:
                    changed = True
                    included.remove(worst_feature)

                if not changed:
                    break
            return included

        def depth_first_selectedindex_fit(self,mineral_type=None, region=None):
            self.technique = "Depth First Search"
            pixel_samples = self.generate_pixel_samples(mineral_type=mineral_type,region=region)
            self.num_pixels = len(pixel_samples)

            self.mineral_data = defaultdict(list)
            self.pixel_y_data = {}
            self.rmse_list = []
            self.adjusted_r_squared_list = []
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
                self.y = self.y[:50]
                self.pixel_y_data[y_index] = self.y 

                self.X = self.spectral_library.T
                self.modelselect = self.depth_first_search(y_index, 0.05)
                self.model_X = self.X[:, list(self.modelselect)]

                self.model_coefficients, _ = nnls(self.model_X, self.y)

                # Find nonzero indicies, names and abundances                 
                self.non_zero_indices = [index for index, coefficient in zip(self.modelselect, self.model_coefficients) if coefficient != 0]
                self.non_zero_spectral_names = [self.spectra_names[index] for index in self.non_zero_indices]
                self.non_zero_coefficients = [coefficient for index, coefficient in zip(self.modelselect, self.model_coefficients) if coefficient != 0]

                # Calculate y_infer and RMSE
                y_infer = np.dot(self.X[:, self.non_zero_indices], self.non_zero_coefficients)
                pixel_rmse = np.sqrt(mean_squared_error(self.y, y_infer))
                
                # Calculate Adjusted rsquared (was not incorporated in results)
                r_squared = 1 - (sum((self.y - y_infer)**2)/sum((self.y-np.mean(self.y))**2))
                n = len(self.y)  
                p = len(self.non_zero_coefficients)  
                adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

                end_time = time.time()
                elapsed_time = end_time - start_time  

                # Save information from each pixel
                self.mineral_data[f'{pixel_sample}'] = list(zip(self.non_zero_spectral_names, self.non_zero_coefficients))
                self.rmse_list.append(pixel_rmse)
                self.adjusted_r_squared_list.append(adjusted_r_squared)
                self.computation_time.append(elapsed_time)
                self.model_size.append(len(self.non_zero_coefficients))

                # Count the number of target mineral - used to find the percent detection for the technique
                if any(keyword in name.lower() for keyword in keywords for name in self.non_zero_spectral_names):
                    inclusion_count += 1 
                     
            # Calculate pixel averages 
            self.rmse_mean = np.mean(self.rmse_list)
            self.rmse_std = np.std(self.rmse_list)
            self.adjusted_r_squared_mean = np.mean(self.adjusted_r_squared_list)
            self.adjusted_r_squared_std = np.std(self.adjusted_r_squared_list)  

            self.computation_time_mean = np.mean(self.computation_time)
            self.model_size_mean = np.mean(self.model_size)
            
            # Calculate target detection rate
            self.target_mineral_proportion = inclusion_count / self.num_pixels if self.num_pixels > 0 else 0  
        
            print(f"Number of pixel samples analyzed: {self.num_pixels}")
            print(f"Number of models including the target mineral: {inclusion_count}")
            print(f"Proportion of models including the target mineral: {self.target_mineral_proportion:.4f}")        
        
            self.plot_median_rmse_spectrum()
            self.plot_metrics_distributions()

        def breadth_first_selectedindex_fit(self, mineral_type=None, region=None):
            self.technique = "Breadth First Search"
            pixel_samples = self.generate_pixel_samples(mineral_type=mineral_type,region=region)
            self.num_pixels = len(pixel_samples)

            self.mineral_data = defaultdict(list)
            self.pixel_y_data = {}
            self.rmse_list = []
            self.adjusted_r_squared_list = []
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
                self.y = self.y[:50]
                self.pixel_y_data[y_index] = self.y 

                self.X = self.spectral_library.T
                self.modelselect = self.breadth_first_search(y_index, 0.05)
                self.model_X = self.X[:, list(self.modelselect)]

                self.model_coefficients, _ = nnls(self.model_X, self.y)

                # Find nonzero indicies, names and abundances 
                self.non_zero_indices = [index for index, coefficient in zip(self.modelselect, self.model_coefficients) if coefficient != 0]
                self.non_zero_spectral_names = [self.spectra_names[index] for index in self.non_zero_indices]
                self.non_zero_coefficients = [coefficient for index, coefficient in zip(self.modelselect, self.model_coefficients) if coefficient != 0]
                
                # Calculate y_infer and RMSE
                y_infer = np.dot(self.X[:, self.non_zero_indices], self.non_zero_coefficients)
                pixel_rmse = np.sqrt(mean_squared_error(self.y, y_infer))
                
                end_time = time.time()
                elapsed_time = end_time - start_time 

                # Save information from each pixel
                self.mineral_data[f'{pixel_sample}'] = list(zip(self.non_zero_spectral_names, self.non_zero_coefficients))
                self.rmse_list.append(pixel_rmse)
                self.computation_time.append(elapsed_time)
                self.model_size.append(len(self.non_zero_coefficients))

                if any(keyword in name.lower() for keyword in keywords for name in self.non_zero_spectral_names):
                    inclusion_count += 1  

            # Calculate pixel averages      
            self.rmse_mean = np.mean(self.rmse_list)
            self.rmse_std = np.std(self.rmse_list) 
            self.computation_time_mean = np.mean(self.computation_time)
            self.model_size_mean = np.mean(self.model_size)

            # Calculate target detection rate
            self.target_mineral_proportion = inclusion_count / self.num_pixels if self.num_pixels > 0 else 0  
        
            print(f"Number of pixel samples analyzed: {self.num_pixels}")
            print(f"Number of models including the target mineral: {inclusion_count}")
            print(f"Proportion of models including the target mineral: {self.target_mineral_proportion:.4f}")    
        
            self.plot_median_rmse_spectrum()
            self.plot_metrics_distributions()
        
        def plot_median_rmse_spectrum(self):
            # Find the median y_inferred spectrum based on RMSE
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


            # Plot the observed spectra
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
             # Plot the distribution of RMSE: Average and Adjusted R-squared (was not incorporated in results)
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