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
                if region == 1:
                    self.ROI = ["alunite hill 1"]
                elif region == 2:
                    self.ROI = ["alunite hill 2"]
                else:
                    self.ROI = ["alunite hill 3"]
            elif mineral_type == 2:
                if region == 1:
                    self.ROI = ["montmorillonite hill 1"]
                elif region == 2:
                    self.ROI = ["montmorillonite hill 2"]
                elif region == 3:
                    self.ROI = ["montmorillonite hill 3"]
                elif region == 4:
                    self.ROI = ["montmorillonite hill 4"]
                else:
                    self.ROI = ["montmorillonite hill 5"]
            elif mineral_type == 3:
                if region == 1:
                    self.ROI = ["kaolinite region 1"]
                else:
                    self.ROI = ["kaolinite region 2"]
            else:
                if region == 1:
                    self.ROI = ["hydrated silica 1"]
                elif region == 2:
                    self.ROI = ["hydrated silica 2"]
                else:
                    self.ROI = ["hydrated silica 3"]

            # Use the isin method for the comparison
            samples = self.df[self.df['Name'].isin(self.ROI)].iloc[:, [2, 3]].values.tolist()
            print(self.ROI)
            return samples
              
        def selectedindex_fit(self, mineral_type=None, region=None, **kwargs):
            self.technique = "BMA"
            start_time = time.time()
            pixel_samples = self.generate_pixel_samples(mineral_type=mineral_type,region=region)
            
            self.mineral_data = defaultdict(list)
            self.pixel_y_data = {}
            for pixel_sample in pixel_samples: 
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

                for self.num_elements in range(1, self.MaxVars + 1):
                    self.model_index_set = None
                    iteration_max_likelihood = 0
                    self.model_index_list = []
                    self.model_likelihood = [] 
                    
                    for model_combination in current_model_set:
                        
                        model_X = self.X[:, list(model_combination)]
                        model_coefficients, _ = nnls(model_X, self.y)

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

                self.probabilities = self.likelihoods / self.likelihood_sum
                self.coefficients = self.coefficients / self.likelihood_sum 

                for name, prob, coef in zip(self.spectra_names, self.probabilities, self.coefficients):
                    if prob > .1:
                        self.mineral_data[y_index].append((name, prob, coef))
            self.final_summary()

            # Uncomment these lines if you want to visualize results
            # f, axarr = plt.subplots(1, 2, figsize=(10, 10))
            # axarr[0].imshow(self.show_image(pixel_samples))
            # axarr[1].imshow(self.show_paper_im()) 

            self.plot_results() 

            end_time = time.time()  # End the timer
            self.elapsed_time = end_time - start_time  
            print(f"Run time: {self.elapsed_time} seconds")   
            return self

        
        def final_summary(self):
            pd.set_option('display.width', 1000)
            average_abundances, average_probability = self.count_mineral()

            indices = [self.spectra_names.index(mineral) for mineral in average_abundances.keys()]            
            self.top_spectra = self.spectral_library.T[:, indices]
            average_abundances_values = np.array(list(average_abundances.values()))
            self.y_infer = np.dot(self.top_spectra, average_abundances_values)   

            # Calculate RMSE for each pixel sample
            rmse_values = {}
            for pixel_sample, observed_spectrum in self.pixel_y_data.items():
                rmse = mean_squared_error(observed_spectrum, self.y_infer, squared=False)
                rmse_values[pixel_sample] = rmse
            min_rmse_pixel = min(rmse_values, key=rmse_values.get)
            min_rmse_spectrum = self.pixel_y_data[min_rmse_pixel]
            self.min_rmse = rmse_values[min_rmse_pixel]
            
            # Print Summary
            data = {'Name': [], 'Category': [], 'Formula': [], 'Probability': [], 'Abundance': []}
            print(f"{self.technique}-Top {self.top_n} most common minerals and their associated most common abundances:")
            for mineral, abundance in average_abundances.items():
                mineral_row = self.chemicaldf[self.chemicaldf['Name']==mineral.split()[1]].iloc[0:1]
                mineral_category = mineral_row.iloc[0]['Category']
                mineral_formula = mineral_row.iloc[0]['Formula']
                probability = average_probability[mineral]

                data['Name'].append(mineral)
                data['Category'].append(mineral_category)
                data['Formula'].append(mineral_formula)
                data['Probability'].append(probability)
                data['Abundance'].append(abundance)
            
            df = pd.DataFrame(data)
            print(df)

            self.plot_spectra_with_lowest_rmse(self.min_rmse, min_rmse_spectrum, self.y_infer, average_abundances)
            return self
        
        def count_mineral(self):
            counts = [len(abundances) for abundances in self.mineral_data.values()]
            self.top_n = statistics.mode(counts)

            # Count the occurrences of each mineral
            mineral_counts = {}
            for abundances in self.mineral_data.values():
                for mineral_abundance in abundances:
                    mineral_name = mineral_abundance[0]
                    if mineral_name not in mineral_counts:
                        mineral_counts[mineral_name] = 0
                    mineral_counts[mineral_name] += 1

            # Get the top n most common minerals
            top_minerals = sorted(mineral_counts, key=mineral_counts.get, reverse=True)[:self.top_n]            

            # Populate the dictionary with abundances and probabilities for each mineral
            mineral_abundances = {mineral: [] for mineral in top_minerals}
            mineral_probabilities = {mineral: [] for mineral in top_minerals}
            for abundances in self.mineral_data.values():
                for mineral_abundance in abundances:
                    mineral_name, probability, coefficient = mineral_abundance
                    if mineral_name in mineral_abundances:
                        mineral_abundances[mineral_name].append(coefficient)
                        mineral_probabilities[mineral_name].append(probability)

            # Initialize a dictionary to store the most common abundance and probability for each top mineral
            average_abundances = {}
            average_probabilities = {}
            for mineral, abundances in mineral_abundances.items():
                average_abundance = sum(abundances) / len(abundances)
                average_abundances[mineral] = average_abundance

            for mineral, probabilities in mineral_probabilities.items():
                average_probability = sum(probabilities) / len(probabilities)
                average_probabilities[mineral] = average_probability

            return average_abundances, average_probabilities
        
        def plot_spectra(self, wavelengths, original_spectrum, top_spectra, top_coefficients, pixel_sample):
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, original_spectrum, label='Original Pixel Spectrum')
            for i, spectrum in enumerate(top_spectra[:self.top_n].T):
                plt.plot(wavelengths, spectrum * top_coefficients[i], label=f'Top Spectrum {i+1}')
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.title(f'Pixel {pixel_sample} and Top {self.top_n} Spectra')
            plt.legend()
            plt.show()

        def plot_spectra_with_lowest_rmse(self, min_rmse, min_rmse_spectrum, y_infer, average_abundances):
            plt.figure(figsize=(12, 6))
            plt.plot(self.wavelengths, min_rmse_spectrum, label='Observed Spectrum with Lowest RMSE', linewidth=3)
            plt.plot(self.wavelengths, y_infer, label='Inferred Spectrum', linestyle='--', linewidth=3, c='black')
            for mineral in average_abundances.keys():
                index = self.spectra_names.index(mineral)
                plt.plot(self.wavelengths, self.spectral_library[index], label=f'{mineral}', alpha=0.5)
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.title(f'{self.technique} \n RMSE: {min_rmse:.2f}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
        
        def plot_results(self):
            plt.figure(figsize=(10, 6))    
            for pixel_sample, observed_spectrum in self.pixel_y_data.items():
                #min_value, max_value = observed_spectrum.min(), observed_spectrum.max()
                #observed_spectrum = -1 + (observed_spectrum - min_value) * 2 / (max_value - min_value)                
                plt.plot(self.wavelengths, observed_spectrum)            
            plt.plot(self.wavelengths, self.y_infer, label='Inferred', linestyle='--', linewidth=2,c='black')
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.legend()
            plt.title(f'{self.technique}')
            plt.grid(True)
            plt.show()

        def show_paper_im(self):
            im = img.imread('img_cuperite_paper.png') 
            return im

        def show_image(self, pixel_samples=[]):
            plt.title("File_Image")
            skip = int(self.n_bands / 4)
            imRGB = np.zeros((self.n_rows, self.n_cols, 3))
            for i in range(3):
                imRGB[:, :, i] = self.stretch(self.image_arr[:, :, i * skip])
            
            # Highlight each pixel sample in the grid
            k = 1
            for loc in pixel_samples:
                x, y = loc                
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        new_y = y + i
                        new_x = x + j
                        if 0 <= new_y < self.n_rows and 0 <= new_x < self.n_cols:
                            imRGB[new_y, new_x, 0] = 0
                            imRGB[new_y, new_x, 1] = 0
                            imRGB[new_y, new_x, 2] = k * 0.1
                k += 1            
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
            
            

             

            
            

             

            
            


            
