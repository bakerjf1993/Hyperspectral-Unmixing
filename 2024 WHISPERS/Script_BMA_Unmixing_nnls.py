import numpy as np 
import pandas as pd 
import spectral.io.envi as envi
from itertools import combinations
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from itertools import chain
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from collections import Counter
import statistics
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.image as img
from sklearn.decomposition import PCA
import time


if __name__ == '__main__':
    print("Running BMA_Umixing_nnls.py. Run from 'Run' notebook")
else:
    
    class BMA:
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
              
        def selectedindex_fit(self, mineral_type=None, region=None, **kwargs):
            self.technique = "BMA"
            pixel_samples = self.generate_pixel_samples(mineral_type=mineral_type,region=region)
            self.num_pixels = len(pixel_samples)

            self.mineral_data = defaultdict(list)
            self.non_zero_spectral_names = []
            self.pixel_y_data = {}
            self.rmse_list = []
            self.adjusted_r_squared_list = []
            self.computation_time = []
            self.model_size = []
            inclusion_count = 0
            
            for pixel_sample in pixel_samples: 
                start_time = time.time()
                y_index = tuple(pixel_sample)
                self.y = self.df[(self.df.iloc[:, 2] == y_index[0]) & (self.df.iloc[:, 3] == y_index[1])].iloc[:, 4:].values.flatten()
                self.pixel_y_data[y_index] = self.y

                self.X = self.spectral_library.T

                self.nRows, self.nCols = np.shape(self.X)
                self.likelihoods = np.zeros(self.nCols)
                self.coefficients = np.zeros(self.nCols)
                self.probabilities = np.zeros(self.nCols)

                if 'MaxVars' in kwargs.keys():
                    self.MaxVars = kwargs['MaxVars']
                else:
                    self.MaxVars = self.nCols

                if 'Priors' in kwargs.keys():
                    if np.size(kwargs['Priors']) == self.nCols:
                        self.Priors = kwargs['Priors']
                    else:
                        print("WARNING: Provided priors error. Using equal priors instead.")
                        print("The priors should be a numpy array of length equal to the number of regressor variables.")
                        self.Priors = np.ones(self.nCols)
                else:
                    self.Priors = np.ones(self.nCols)

                self.likelihood_sum = 0
                self.max_likelihood = 0
                self.num_elements = 1
                self.best_model_index_set = None
                candidate_models = list(range(self.nCols))
                current_model_set = list(combinations(candidate_models, self.num_elements)) 

                # Sets the maximuum number variables of we plan to explore
                for self.num_elements in range(1, self.MaxVars + 1):
                    self.model_index_set = None
                    iteration_max_likelihood = 0
                    self.model_index_list = []
                    self.model_likelihood = [] 
                    
                    # We are iterating over every possible model combination up to max vars
                    for model_combination in current_model_set:
                        
                        model_X = self.X[:, list(model_combination)]
                        model_coefficients, _ = nnls(model_X, self.y)

                        # Calculates the model likelihood
                        rss = np.sum((self.y - np.dot(model_X, model_coefficients)) ** 2)
                        k = model_X.shape[1]  
                        n = self.y.shape[0]
                        bic = n * np.log(rss / n) + k * np.log(n)
                        model_likelihood = np.exp(-bic / 2) * np.prod(self.Priors[list(model_combination)])
                        
                        
                        if model_likelihood > iteration_max_likelihood:
                            iteration_max_likelihood = model_likelihood
                            self.model_index_set = model_combination
                            self.model_set_coefficients = model_coefficients
                                        
                        self.likelihood_sum += model_likelihood                  

                        self.model_index_list.append(model_combination)
                        self.model_likelihood.append(model_likelihood)
                        for i, model_idx in enumerate(model_combination):
                            self.likelihoods[model_idx] += model_likelihood
                            self.coefficients[model_idx] += model_coefficients[i] * model_likelihood
                    
                    self.model_probability = np.asarray(self.model_likelihood) / self.likelihood_sum

                    if iteration_max_likelihood > self.max_likelihood:
                        self.max_likelihood = iteration_max_likelihood                                  

                    # Sets a threshold for which models/model pairs to include for the next iteration
                    top_models_threshold = round(0.05 * self.max_likelihood)
                    candidate_models = []
                    current_model_set = []
                    for i, (model_idx, model_likelihood) in enumerate(zip(self.model_index_list, self.model_likelihood)):
                        if model_likelihood > top_models_threshold:
                            for idx in range(self.nCols):
                                current_model_set.append(model_idx + (idx,) if model_idx else (idx,))                  
                                                    
                    #if top_models_threshold < self.num_elements + 1 or len(current_model_set) == 0:
                    if len(current_model_set) == 0:
                        print("The number of variables required for the next iteration exceed the number of candidate models")
                        print(f"BMA is finishing early at iteration: {self.num_elements}")
                        break

                # Calculates the average model probabilities and coefficients
                self.probabilities = self.likelihoods / self.likelihood_sum
                self.coefficients = self.coefficients / self.likelihood_sum 
 

                y_infer = np.zeros_like(self.y)

                p = 0 # number of features 
                # Detemines the most likely features
                for name, prob, coef in zip(self.spectra_names, self.probabilities, self.coefficients):
                    if prob > .1:
                        index = self.spectra_names.index(name)
                        spectrum = self.spectral_library[index]
                        y_infer += coef * spectrum # Calculate y_infer 
                        p += 1
                        self.non_zero_spectral_names.append(name)                

                # Calculate RMSE
                pixel_rmse = np.sqrt(mean_squared_error(self.y, y_infer))

                # Calculate Adjusted rsquared (was not incorporated in results)
                r_squared = 1 - (sum((self.y - y_infer)**2)/sum((self.y-np.mean(self.y))**2))
                n = len(self.y)                   
                adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)                      
                                        
                end_time = time.time()
                elapsed_time = end_time - start_time 

                # Save information from each pixel
                self.mineral_data[y_index].append((y_infer, pixel_rmse))
                self.rmse_list.append(pixel_rmse)
                self.adjusted_r_squared_list.append(adjusted_r_squared)
                self.computation_time.append(elapsed_time)
                self.model_size.append(p)

                if any(self.target_mineral.lower() in name.lower() for name in self.non_zero_spectral_names):
                    inclusion_count += 1  

            # Calculate other pixel averages   
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

            median_data = self.mineral_data[median_pixel_sample][0]
            median_inferred_spectrum = median_data[0]
            
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
            plt.figure(figsize=(13, 5))
            plt.suptitle(self.technique)

            # RMSE Distribution
            plt.subplot(1, 2, 1)
            plt.hist(self.rmse_list, bins=20, color='skyblue', edgecolor='black')
            plt.title('Distribution of RMSE')
            plt.xlabel('RMSE')
            plt.ylabel('Frequency')
            # Add mean and variance as text annotations
            plt.text(0.95, 0.85, f"Mean: {self.rmse_mean:.4f}\nStandard Deviation: {self.rmse_std:.4f}", 
                    transform=plt.gca().transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            # Adjusted R-squared Distribution (was not incorporated in results)
            plt.subplot(1, 2, 2)
            plt.hist(self.adjusted_r_squared_list, bins=20, color='lightcoral', edgecolor='black')
            plt.title('Distribution of Adjusted R-squared')
            plt.xlabel('Adjusted R-squared')
            plt.ylabel('Frequency')
            # Add mean and variance as text annotations
            plt.text(0.95, 0.85, f"Mean: {self.adjusted_r_squared_mean:.4f}\nStandard Deviation: {self.adjusted_r_squared_std:.4f}", 
                    transform=plt.gca().transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

            plt.tight_layout()
            plt.show()