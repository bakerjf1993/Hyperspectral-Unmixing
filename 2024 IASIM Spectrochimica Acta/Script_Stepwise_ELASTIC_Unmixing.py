import numpy as np 
import pandas as pd 
import spectral.io.envi as envi
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from itertools import chain
from scipy.stats import t
from sklearn.linear_model import ElasticNet
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
    print("Running Stepwise_ELASTIC_Unmixing.py. Run from 'Run' notebook")
else:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    class STEPWISE:
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
                self.ROI_name = "Alunite"
                if region == 1:
                    self.ROI = ["alunite hill 1"]
                elif region == 2:
                    self.ROI = ["alunite hill 2"]
                else:
                    self.ROI = ["alunite hill 3"]
            elif mineral_type == 2:
                self.ROI_name = "Montmorillonite"
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
                self.ROI_name = "Kaolinite"
                if region == 1:
                    self.ROI = ["kaolinite region 1"]
                else:
                    self.ROI = ["kaolinite region 2"]
            elif mineral_type == 4:
                self.ROI_name = "hydrated silica"
                if region == 1:
                    self.ROI = ["hydrated silica 1"]
                elif region == 2:
                    self.ROI = ["hydrated silica 2"]
                else:
                    self.ROI = ["hydrated silica 3"]
            else:
                self.ROI_name = "montmorillonite"
                if region == 1:
                    self.ROI = ["montmorillonite 1"]
                elif region == 2:
                    self.ROI = ["montmorillonite 2"]
                elif region == 3:
                    self.ROI = ["montmorillonite 3"]
                elif region == 4:
                    self.ROI = ["montmorillonite 4"]
                else:
                    self.ROI = ["montmorillonite 5"]

            # Use the isin method for the comparison
            samples = self.df[self.df['Name'].isin(self.ROI)].iloc[:, [2, 3]].values.tolist()
            print(self.ROI)
            return samples  

        def forward_regression(self, y_index, threshold_in):
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
        
        def backward_regression(self, y_index, threshold_out):
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

        def forward_selectedindex_fit(self,mineral_type=None, region=None):
            self.technique = "Forward Stepwise - Elasticnet Regression"
            start_time = time.time()
            pixel_samples = self.generate_pixel_samples(mineral_type=mineral_type, region=region)
            self.num_pixels = len(pixel_samples)

            self.mineral_data = defaultdict(list)
            self.pixel_y_data = {}
            for pixel_sample in pixel_samples: 
                # Obtain observed spectra from samples and store for later use               
                y_index = tuple(pixel_sample)
                self.y = self.df[(self.df.iloc[:, 2] == y_index[0]) & (self.df.iloc[:, 3] == y_index[1])].iloc[:, 4:].values.flatten()
                self.pixel_y_data[y_index] = self.y

                self.X = self.spectral_library.T
                self.modelselect = self.forward_regression(y_index, 0.05)
                self.model_X = self.X[:, list(self.modelselect)]

                # Fit the spectral library to each pixel sample
                elastc_model = ElasticNet(alpha=.0001, positive=True)
                elastc_model.fit(self.model_X, self.y)
                self.model_coefficients = elastc_model.coef_
                                
                self.non_zero_indices = [index for index, coefficient in zip(self.modelselect, self.model_coefficients) if coefficient != 0]
                self.non_zero_spectral_names = [self.spectra_names[index] for index in self.non_zero_indices]
                self.non_zero_coefficients = [coefficient for index, coefficient in zip(self.modelselect, self.model_coefficients) if coefficient != 0]

                
                self.mineral_data[f'{pixel_sample}'] = list(zip(self.non_zero_spectral_names, self.non_zero_coefficients))            
            
            self.final_summary()

            f, axarr = plt.subplots(1,2,figsize=(10, 10))
            axarr[0].imshow(self.show_image(pixel_samples))
            axarr[1].imshow(self.show_paper_im()) 

            self.plot_results()
            end_time = time.time()  # End the timer
            self.elapsed_time = end_time - start_time  
            print(f"Run time: {self.elapsed_time} seconds")     
            return self

        def backward_selectedindex_fit(self, mineral_type=None, region=None):
            self.technique = "Backward Stepwise Regression"
            start_time = time.time()
            pixel_samples = self.generate_pixel_samples(mineral_type=mineral_type, region=region)
            
            self.mineral_data = defaultdict(list)
            self.pixel_y_data = {}
            for pixel_sample in pixel_samples: 
                # Obtain observed spectra from samples and store for later use               
                y_index = tuple(pixel_sample)
                self.y = self.df[(self.df.iloc[:, 2] == y_index[0]) & (self.df.iloc[:, 3] == y_index[1])].iloc[:, 4:].values.flatten()
                self.pixel_y_data[y_index] = self.y

                self.X = self.spectral_library.T
                self.modelselect = self.backward_regression(y_index, 0.05)
                self.model_X = self.X[:, list(self.modelselect)]

                # Fit the spectral library to each pixel sample
                elastc_model = ElasticNet(alpha=.001, positive=True)
                elastc_model.fit(self.model_X, self.y)
                self.model_coefficients = elastc_model.coef_

                
                self.non_zero_indices = [index for index, coefficient in zip(self.modelselect, self.model_coefficients) if coefficient != 0]
                self.non_zero_spectral_names = [self.spectra_names[index] for index in self.non_zero_indices]
                self.non_zero_coefficients = [coefficient for index, coefficient in zip(self.modelselect, self.model_coefficients) if coefficient != 0]
                
                self.mineral_data[f'{pixel_sample}'] = list(zip(self.non_zero_spectral_names, self.non_zero_coefficients))            
            
            self.final_summary()

            f, axarr = plt.subplots(1,2,figsize=(10, 10))
            axarr[0].imshow(self.show_image(pixel_samples))
            axarr[1].imshow(self.show_paper_im()) 

            self.plot_results()
            end_time = time.time()  # End the timer
            self.elapsed_time = end_time - start_time  
            print(f"Run time: {self.elapsed_time} seconds")     
            return self
            
        
        def final_summary(self):
            pd.set_option('display.width', 1000)
            # Determine the number of top minerals based on the most common minerals in the model             
            average_abundances = self.count_mineral()

            # Calculate y_infer
            indices = [self.spectra_names.index(mineral) for mineral in average_abundances.keys()]            
            self.top_spectra = self.spectral_library.T[:, indices]
            average_abundances_values = np.array(list(average_abundances.values()))
            self.y_infer = np.dot(self.top_spectra, average_abundances_values)   
            
            rmse_values = {}
            for pixel_sample, observed_spectrum in self.pixel_y_data.items():
                rmse = mean_squared_error(observed_spectrum, self.y_infer, squared=False)
                rmse_values[pixel_sample] = rmse
            min_rmse_pixel = min(rmse_values, key=rmse_values.get)
            min_rmse_spectrum = self.pixel_y_data[min_rmse_pixel]
            self.min_rmse = rmse_values[min_rmse_pixel]
            
            # Print the top n most common minerals and their associated most common abundances
            data = {'Name': [], 'Category': [], 'Formula': [], 'Abundance': []}
            print(f"{self.technique}-Top {self.top_n} most common minerals and their associated most common abundances:")
            for mineral, abundance in average_abundances.items():
                mineral_row = self.chemicaldf[self.chemicaldf['Name']==mineral.split()[1]].iloc[0:1]
                mineral_category = mineral_row.iloc[0]['Category']
                mineral_formula = mineral_row.iloc[0]['Formula']
                data['Name'].append(mineral)
                data['Category'].append(mineral_category)
                data['Formula'].append(mineral_formula)
                data['Abundance'].append(abundance)
            
            df = pd.DataFrame(data)
            print(df)

            self.plot_spectra_with_lowest_rmse(self.min_rmse, min_rmse_spectrum, self.y_infer, average_abundances)
            return self
        
        def count_mineral(self):
            counts = [len(abundances) for abundances in self.mineral_data.values()]
            #self.top_n = statistics.mode(counts)
            self.top_n = round(sum(counts)/len(counts))

            # Count the occurrences of each mineral
            mineral_counts = {}            
            accuracy_count = 0 
            for abundances in self.mineral_data.values():
                loop_count = 0
                for mineral_abundance in abundances:
                    mineral_name = mineral_abundance[0]                    
                    if mineral_name not in mineral_counts:
                        mineral_counts[mineral_name] = 0
                    mineral_counts[mineral_name] += 1
                    if self.ROI_name in mineral_name:
                        if loop_count < 1:
                            accuracy_count += 1
                            loop_count += 1
                
            self.accuracy_level = accuracy_count/self.num_pixels * 100
            print(f"Percent Accracy: {round(self.accuracy_level)}%")

            # Get the top n most common minerals
            top_minerals = sorted(mineral_counts, key=mineral_counts.get, reverse=True)[:self.top_n]

            # Populate the dictionary with abundances for each mineral
            mineral_abundances = {mineral: [] for mineral in top_minerals}
            for abundances in self.mineral_data.values():
                for mineral_abundance in abundances:
                    mineral_name, abundance = mineral_abundance
                    if mineral_name in mineral_abundances:
                        mineral_abundances[mineral_name].append(abundance)

            # Initialize a dictionary to store the average abundance for each top mineral
            average_abundances = {}
            for mineral, abundances in mineral_abundances.items():
                average_abundance = sum(abundances) / len(abundances)
                average_abundances[mineral] = average_abundance

            return average_abundances
        
        def plot_spectra_with_lowest_rmse(self, min_rmse, min_rmse_spectrum, y_infer, average_abundances):
            plt.figure(figsize=(10, 6))
            plt.plot(self.wavelengths, min_rmse_spectrum, label='Observed Spectrum with Lowest RMSE', linewidth=3)
            plt.plot(self.wavelengths, y_infer, label='Inferred Spectrum', linestyle='--', linewidth=3, c='black')
            for mineral in average_abundances.keys():
                index = self.spectra_names.index(mineral)
                plt.plot(self.wavelengths, self.spectral_library[index], label=f'{mineral}', alpha=0.5)
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.title(f'{self.technique} \n RMSE: {min_rmse:.2f}')
            plt.legend(ncol=2,bbox_to_anchor=(.95, -0.15))
            plt.gcf().patch.set_alpha(0)
            plt.show()
        
        def plot_results(self):
            plt.figure(figsize=(10, 6))    
            for pixel_sample, observed_spectrum in self.pixel_y_data.items():
                plt.plot(self.wavelengths, observed_spectrum)            
            plt.plot(self.wavelengths, self.y_infer, label='Inferred', linestyle='--', linewidth=2, c='black')
            plt.xlabel('Wavelength')
            plt.ylabel('Intensity')
            plt.legend()
            plt.title(f'{self.technique}')
            plt.grid(True)
            plt.gcf().patch.set_alpha(0)
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
            
            

             

            
            


            

        
            
            

             

            
            


            
