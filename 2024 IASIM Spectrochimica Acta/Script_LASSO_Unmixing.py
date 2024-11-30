import numpy as np 
import pandas as pd 
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import time

import warnings
from sklearn.exceptions import ConvergenceWarning


if __name__ == '__main__':
    print("Running LASSO_Unmixing.py. Run from 'Run' notebook")
else:
    
    class LASSO:
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
              
        def selectedindex_fit(self, mineral_type=None, region=None):
            self.technique = "LASSO Regression"
           
            pixel_samples = self.generate_pixel_samples(mineral_type=mineral_type,region=region)
            self.num_pixels = len(pixel_samples)

            self.mineral_data = defaultdict(list)
            self.pixel_y_data = {}
            self.rmse_list = []
            self.adjusted_r_squared_list = []
            self.y_infer_data= []
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

                # Fit the spectral library to each pixel sample
                lasso_model = Lasso(alpha=.0004, positive=True)
                #print([len(self.X), len(self.y)])    
                lasso_model.fit(self.X, self.y)
                self.model_coefficients = lasso_model.coef_

                self.non_zero_indices = np.where(self.model_coefficients != 0)[0]
                self.non_zero_coefficients = self.model_coefficients[self.non_zero_indices]
                self.non_zero_spectral_names = [self.spectra_names[index] for index in self.non_zero_indices]

                y_infer = np.dot(self.X[:, self.model_coefficients != 0], self.non_zero_coefficients)
                pixel_rmse = np.sqrt(mean_squared_error(self.y, y_infer))
                
                #r_squared = r2_score(self.y, y_infer)
                r_squared = 1 - (sum((self.y - y_infer)**2)/sum((self.y-np.mean(self.y))**2))
                n = len(self.y)  
                p = len(self.non_zero_coefficients)  
                adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

                end_time = time.time()
                elapsed_time = end_time - start_time                
                
                self.y_infer_data.append(y_infer)
                self.mineral_data[f'{pixel_sample}'] = list(zip(self.non_zero_spectral_names, self.non_zero_coefficients))
                self.rmse_list.append(pixel_rmse)
                self.adjusted_r_squared_list.append(adjusted_r_squared)
                self.computation_time.append(elapsed_time)
                self.model_size.append(len(self.non_zero_coefficients))

                if any(keyword in name.lower() for keyword in keywords for name in self.non_zero_spectral_names):
                    inclusion_count += 1 
                     
            self.rmse_mean = np.mean(self.rmse_list)
            self.rmse_std = np.std(self.rmse_list)
            self.adjusted_r_squared_mean = np.mean(self.adjusted_r_squared_list)
            self.adjusted_r_squared_std = np.std(self.adjusted_r_squared_list)  

            self.computation_time_mean = np.mean(self.computation_time)
            self.model_size_mean = np.mean(self.model_size)   

            total_pixel_samples = len(pixel_samples)
            self.target_mineral_proportion = inclusion_count / total_pixel_samples if total_pixel_samples > 0 else 0       
        
            print(f"Number of pixel samples analyzed: {total_pixel_samples}")
            print(f"Average unmixing computation time: {self.computation_time_mean}")
            print(f"Average RMSE: {self.rmse_mean}")
            print(f"Number of models including the target mineral: {inclusion_count}")
            print(f"Proportion of models including the target mineral: {self.target_mineral_proportion:.4f}")
            
            self.plot_mean_rmse_spectrum() 
            self.final_summary()
            self.plot_results()   
        
        def plot_mean_rmse_spectrum(self):
            # Find the pixel sample with the closest RMSE to the mean RMSE
            mean_rmse = np.mean(self.rmse_list)
            closest_index_mean = np.argmin(np.abs(np.array(self.rmse_list) - mean_rmse))

            inferred_spectrum = self.y_infer_data[closest_index_mean]
            closest_pixel = list(self.pixel_y_data.keys())[closest_index_mean]
            observed_spectrum = self.pixel_y_data[closest_pixel]

            # Retrieve the contributing spectra and coefficients for the closest pixel
            lasso_model = Lasso(alpha=.0004, positive=True)
            lasso_model.fit(self.spectral_library.T, observed_spectrum)
            model_coefficients = lasso_model.coef_

            non_zero_indices = np.where(model_coefficients != 0)[0]
            non_zero_coefficients = model_coefficients[non_zero_indices]
            non_zero_spectral_names = [self.spectra_names[i] for i in non_zero_indices]

            # Plot the observed and inferred spectra
            plt.figure(figsize=(12, 8))
            plt.plot(self.wavelengths, observed_spectrum, label="Observed Spectrum", linewidth=3)
            plt.plot(self.wavelengths, inferred_spectrum, label="Inferred Spectrum (Mean RMSE)", linestyle="--", linewidth=3, color="black")

            # Add individual contributing spectra scaled by their coefficients
            for i, spectrum_index in enumerate(non_zero_indices):
                scaled_spectrum = self.spectral_library[spectrum_index, :] * non_zero_coefficients[i]
                plt.plot(self.wavelengths, scaled_spectrum, label=f"{non_zero_spectral_names[i]} (x{non_zero_coefficients[i]:.2f})", alpha=0.5)

            plt.suptitle(self.technique)
            plt.title(f"Observed vs. Inferred Spectrum (Mean RMSE: {mean_rmse:.4f})", fontsize=14)
            plt.xlabel("Wavelength", fontsize=12)
            plt.ylabel("Intensity", fontsize=12)
            plt.legend(ncol=2,bbox_to_anchor=(.95, -0.15))
            plt.grid(True)
            plt.show()

            data = []
            for name, coefficient in zip(non_zero_spectral_names, non_zero_coefficients):
                # Find the category and formula for the mineral
                match = self.chemicaldf[self.chemicaldf['Name']==name.split()[1]].iloc[0:1]
                if not match.empty:
                    category = match.iloc[0]['Category']
                    formula = match.iloc[0]['Formula']
                else:
                    category = "Unknown"
                    formula = "Unknown"

                # Append to the table data
                data.append([name, category, formula, round(coefficient, 4)])

            # Create a DataFrame for the table
            mineral_table = pd.DataFrame(data, columns=["Mineral", "Category", "Formula", "Abundance"])
            mineral_table = mineral_table.sort_values(by="Abundance", ascending=False)

            # Print the table
            print("\nMinerals Contributing to Inferred Spectrum:")
            print(mineral_table.to_string(index=False))

            # Display the table as a plot (optional)
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.axis('tight')
            ax.axis('off')
            formatted_data = [
                [
                    row[0] if len(row[0]) <= 60 else f"{row[0][:17]}...",  # Truncate long names
                    row[1],  # Category
                    row[2],  # Formula
                    f"{row[3]:.4f}"  # Ensure rounding in the plot
                ]
                for row in mineral_table.values
            ]

            table = ax.table(
                cellText=formatted_data,
                colLabels=mineral_table.columns,
                cellLoc="center",
                loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.5, 1.5)  # Scale table for larger size

            # Adjust column widths to fit content
            for key, cell in table.get_celld().items():
                cell.set_text_props(ha="center", va="center")  # Center align text

            plt.show()

        def final_summary(self):
            pd.set_option('display.width', 1000)
            # Determine the number of top minerals based on the most common minerals in the model             
            average_abundances = self.count_mineral()

            # Calculate y_infer
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

            ## Count the occurrences of each mineral
            mineral_counts = {} 
            for abundances in self.mineral_data.values():
                for mineral_abundance in abundances:
                    mineral_name = mineral_abundance[0]                    
                    if mineral_name not in mineral_counts:
                        mineral_counts[mineral_name] = 0
                    mineral_counts[mineral_name] += 1
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
            #plt.rcParams.update({'font.size': 20})
            plt.gcf().patch.set_alpha(0)
            plt.show()