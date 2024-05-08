import importlib

import Script_RIDGE_Unmixing
from Script_RIDGE_Unmixing import RIDGE
importlib.reload(Script_RIDGE_Unmixing)

import Script_OLS_Unmixing
from Script_OLS_Unmixing import OLS
importlib.reload(Script_OLS_Unmixing)

import Script_NNLS_Unmixing
from Script_NNLS_Unmixing import NNLS
importlib.reload(Script_NNLS_Unmixing)

import Script_LASSO_Unmixing
from Script_LASSO_Unmixing import LASSO
importlib.reload(Script_LASSO_Unmixing)

import Script_Stepwise_Unmixing
from Script_Stepwise_Unmixing import STEPWISE
importlib.reload(Script_Stepwise_Unmixing)

import Script_BMA_Unmixing_nnls
from Script_BMA_Unmixing_nnls import BMA
importlib.reload(Script_BMA_Unmixing_nnls)

import spectral.io.envi as envi
from sklearn.decomposition import PCA
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as img
from IPython.display import display

import re
from matplotlib.lines import Line2D

if __name__ == '__main__':
    print("Running Compare_Technique.py. Run from 'Run' notebook")
else:
    
    class COMPARE:
        def __init__(self, image_hdr_filename, image_filename, spectral_library_filename):
            self.load_new_image(image_hdr_filename, image_filename)
            self.load_new_spectral_library(spectral_library_filename)
        
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

        def get_marker(self, mineral, mineral_to_search_for):
            if re.search(mineral_to_search_for[:4], mineral):
                return 'o'
            else:
                return 'x'
        
        def compare_tech(self,image_hdr_filename,image_filename,spectral_library_filename,pixel_index,mineral_to_search_for, alpha_lasso=1, alpha_ridge=0.5):            
                        
            ols_unmixer = OLS(image_hdr_filename, image_filename, spectral_library_filename)
            ols_result = ols_unmixer.selectedindex_fit(y_index=pixel_index)
            ols_summary = [(name, coefficient) for name, coefficient in zip(ols_unmixer.non_zero_spectral_names, ols_unmixer.non_zero_coefficients)]
            ols_summary.sort(key=lambda x: x[1], reverse=True)
            ols_summary_filtered = [item for item in ols_summary if item[1] >= 0.1]
            ols_summary = ols_summary_filtered
            print(f"OLS Mixer RMSE: {ols_unmixer.rmse}, ols_summary:{ols_summary}")

            nnls_unmixer = NNLS(image_hdr_filename, image_filename, spectral_library_filename)
            nnls_result = nnls_unmixer.selectedindex_fit(y_index=pixel_index)
            nnls_summary = [(name, coefficient) for name, coefficient in zip(nnls_unmixer.non_zero_spectral_names, nnls_unmixer.non_zero_coefficients)]
            nnls_summary.sort(key=lambda x: x[1], reverse=True)
            nnls_summary_filtered = [item for item in nnls_summary if item[1] >= 0.1]
            nnls_summary = nnls_summary_filtered
            print(f"NNLS Mixer RMSE: {nnls_unmixer.rmse}, nnls_summary:{nnls_summary}")

            lasso_unmixer = LASSO(image_hdr_filename, image_filename, spectral_library_filename)
            lasso_result = lasso_unmixer.selectedindex_fit(y_index=pixel_index, alpha=alpha_lasso)
            lasso_summary = [(name, coefficient) for name, coefficient in zip(lasso_unmixer.non_zero_spectral_names, lasso_unmixer.non_zero_coefficients)]
            lasso_summary.sort(key=lambda x: x[1], reverse=True)
            lasso_summary_filtered = [item for item in lasso_summary if item[1] >= 0.1]
            lasso_summary = lasso_summary_filtered
            print(f"Lasso Mixer RMSE: {lasso_unmixer.rmse}, lasso_summary:{lasso_summary}")

            ridge_unmixer = RIDGE(image_hdr_filename, image_filename, spectral_library_filename)
            ridge_result = ridge_unmixer.selectedindex_fit(y_index=pixel_index, alpha=alpha_ridge)
            ridge_summary = [(name, coefficient) for name, coefficient in zip(ridge_unmixer.non_zero_spectral_names, ridge_unmixer.non_zero_coefficients)]
            ridge_summary.sort(key=lambda x: x[1], reverse=True)
            ridge_summary_filtered = [item for item in ridge_summary if item[1] >= 0.1]
            ridge_summary = ridge_summary_filtered
            print(f"Ridge Mixer RMSE: {ridge_unmixer.rmse},ridge_summary:{ridge_summary}")

            step_wise_unmixer = STEPWISE(image_hdr_filename, image_filename, spectral_library_filename)
            step_forward_result = step_wise_unmixer.forward_selectedindex_fit(y_index=pixel_index)
            step_forward_summary = [(name, coefficient) for name, coefficient in zip(step_wise_unmixer.non_zero_spectral_names, step_wise_unmixer.summary_model_coefficients)]
            step_forward_summary.sort(key=lambda x: x[1], reverse=True)
            step_forward_summary_filtered = [item for item in step_forward_summary if item[1] >= 0.1]
            step_forward_summary = step_forward_summary_filtered
            print(f"Forward_step Mixer RMSE: {step_wise_unmixer.rmse}, step_forward_summary:{step_forward_summary}")

            step_back_result = step_wise_unmixer.backward_selectedindex_fit(y_index=pixel_index)
            step_back_summary = [(name, coefficient) for name, coefficient in zip(step_wise_unmixer.non_zero_spectral_names, step_wise_unmixer.summary_model_coefficients)]
            step_back_summary.sort(key=lambda x: x[1], reverse=True)
            step_back_summary_filtered = [item for item in step_back_summary if item[1] >= 0.05]
            step_back_summary = step_back_summary_filtered
            print(f"Backward_step Mixer RMSE: {step_wise_unmixer.rmse}, step_back_summary:{step_back_summary}")

            BMA_unmixer = BMA(image_hdr_filename, image_filename, spectral_library_filename)
            BMA_result = BMA_unmixer.selectedindex_fit(y_index=pixel_index, MaxVars=5)
            BMA_summary = [(name, probaiilty, coefficient) for name,probaiilty, coefficient in zip(BMA_unmixer.spectra_names, BMA_unmixer.probabilities, BMA_unmixer.coefficients)]
            BMA_summary.sort(key=lambda x: x[1], reverse=True)
            BMA_summary_filtered = [item for item in BMA_summary if item[2] >= 0.1]
            BMA_summary = BMA_summary_filtered
            print(f"BMA Mixer RMSE: {BMA_unmixer.rmse}, BMA_summary:{BMA_summary}")
            
            combined_model_summaries = {}
            combined_model_summaries['OLS'] = (ols_summary, len(ols_summary), ols_unmixer.rmse)
            combined_model_summaries['NNLS'] = (nnls_summary, len(nnls_summary), nnls_unmixer.rmse)
            combined_model_summaries['Lasso'] = (lasso_summary, len(lasso_summary), lasso_unmixer.rmse)
            combined_model_summaries['Ridge'] = (ridge_summary, len(ridge_summary), ridge_unmixer.rmse)
            combined_model_summaries['Stepwise_forward'] = (step_forward_summary, len(step_forward_summary), step_wise_unmixer.rmse)
            combined_model_summaries['Stepwise_backward'] = (step_back_summary, len(step_back_summary), step_wise_unmixer.rmse)
            combined_model_summaries['BMA'] = (BMA_summary, len(BMA_summary), BMA_unmixer.rmse)

            x_vals, y_vals = [], []
            colors = {'OLS': 'blue', 'NNLS': 'green', 'Lasso': 'orange', 'Ridge': 'red', 'Stepwise_forward': 'purple', 'Stepwise_backward': 'brown', 'BMA': 'gray'}
            
            
            
            for model_name, model_data in combined_model_summaries.items():
                summary = model_data[0]
                model_length = model_data[1]
                model_rmse = model_data[2]
                color = colors[model_name]

                               
                
                # If model is BMA, add scatter plots for each mineral
                if model_name == 'BMA':
                    for i, (mineral, alpha, coefficient) in enumerate(summary):
                        noise_x = np.random.normal(0, .6, len(summary))
                        noise_y = np.random.normal(0, 0.002, len(summary))
                        marker = self.get_marker(mineral, mineral_to_search_for)
                        plt.scatter(model_length + noise_x, model_rmse + noise_y, c=color, marker=marker, s=5)
                else: # Otherwise, add scatter plot for the entire summary
                    for i, (mineral, coefficient) in enumerate(summary):
                        noise_x = np.random.normal(0, 0.6, len(summary))
                        noise_y = np.random.normal(0, 0.002, len(summary))
                        x_vals.append(model_length)
                        y_vals.append(model_rmse)
                        marker = self.get_marker(mineral, mineral_to_search_for)
                        plt.scatter(model_length + noise_x, model_rmse + noise_y, c=color, marker=marker, s=5)
            # Plot the data
            plt.xlabel('Number of  Features')
            plt.ylabel('RMSE')
            plt.title('Model Size vs Error Rate')

            # Create a legend
            custom_handles = [Line2D([0], [0], color=color, lw=2) for color in colors.values()]
            plt.legend(custom_handles, colors.keys(), loc='upper right')

            # Display the plot
            plt.show()
            '''for model_name, model_data in combined_model_summaries.items():
                summary = model_data[0]
                for entry in summary:
                    if model_name == 'BMA':
                        mineral, _, _ = entry
                    else:
                        mineral, _ = entry
                    if re.search(mineral_to_search_for[:4], mineral):
                        print(model_name, mineral)'''
            
            
            
            
            
            return self