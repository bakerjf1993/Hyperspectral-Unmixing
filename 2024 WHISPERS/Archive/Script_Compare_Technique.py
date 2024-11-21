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

import Script_BMAquad_Unmixing_nnls
from Script_BMAquad_Unmixing_nnls import BMAquad
importlib.reload(Script_BMAquad_Unmixing_nnls)

import spectral.io.envi as envi
from sklearn.decomposition import PCA
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as img
from IPython.display import display

import re
from matplotlib.lines import Line2D

image_hdr_filename = "cup95eff.hdr"
image_filename = "cup95eff.int"
pixel_location = "ROIs2.csv"
spectral_library_filename = "usgs_minerals.hdr"

if __name__ == '__main__':
    print("Running Compare_Technique.py. Run from 'Run' notebook")
else:
    
    class COMPARE:
        pd.set_option('display.max_rows', 500)
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
        
        def run_techniques_multiple_times(self, mineral_type=None, region=None):
            if mineral_type == 1:
                self.ROI = "Alunite"
            elif mineral_type == 2:
                self.ROI = "Montmorillonite"
            elif mineral_type == 3:
                self.ROI = "Kaolinite"
            else:
                self.ROI = "Hydrated silica"

            techniques = {
                'OLS': OLS,
                'NNLS': NNLS,
                'RIDGE': RIDGE,
                'LASSO': LASSO,
                'STEPWISE': STEPWISE,
                'BMA': BMA,
                'BMAquad': BMAquad
            }
            self.results = []
            
            for technique_name, TechniqueClass in techniques.items():
                technique = TechniqueClass(
                    image_hdr_filename,
                    image_filename,
                    pixel_location,
                    spectral_library_filename
                )
                if technique_name == 'STEPWISE':
                    technique.forward_selectedindex_fit(mineral_type=mineral_type, region=region)                    
                    rmse = technique.min_rmse 
                    model_size = technique.top_n  
                    runtime = technique.elapsed_time                     
                    self.results.append({'technique': technique_name, 'rmse': rmse, 'model_size': model_size, 'runtime':runtime})

                    technique.backward_selectedindex_fit(mineral_type=mineral_type, region=region)                    
                    rmse = technique.min_rmse 
                    model_size = technique.top_n  
                    runtime = technique.elapsed_time                     
                    self.results.append({'technique': technique_name, 'rmse': rmse, 'model_size': model_size, 'runtime':runtime})

                elif technique_name == 'BMA':
                    technique.selectedindex_fit(mineral_type=mineral_type, region=region, MaxVars=2)                        
                    rmse = technique.min_rmse  
                    model_size = technique.top_n  
                    runtime = technique.elapsed_time                         
                    self.results.append({'technique': technique_name, 'rmse': rmse, 'model_size': model_size, 'runtime':runtime})

                elif technique_name == 'BMAquad':
                    technique.selectedindex_fit(mineral_type=mineral_type, region=region, MaxVars=2)                        
                    rmse = technique.min_rmse  
                    model_size = technique.top_n  
                    runtime = technique.elapsed_time                         
                    self.results.append({'technique': technique_name, 'rmse': rmse, 'model_size': model_size, 'runtime':runtime})

                else:
                    technique.selectedindex_fit(mineral_type=mineral_type, region=region)  
                    rmse = technique.min_rmse  
                    model_size = technique.top_n     
                    runtime = technique.elapsed_time                         
                    self.results.append({'technique': technique_name, 'rmse': rmse, 'model_size': model_size, 'runtime':runtime})
            print(self.results)

            self.plot_results()
            
            return self
        
        def plot_results(self):
            colors = {'OLS': 'blue', 'NNLS': 'red', 'RIDGE': 'green', 'LASSO': 'purple', 'STEPWISE': 'orange','BMA':'black','BMAquad':'darkgrey'}
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                                    markerfacecolor=color, markersize=10) for key, color in colors.items()]
            
            for result in self.results:
                plt.scatter(result['runtime'], result['rmse'], 
                            s=result['model_size']*10,  # Adjust size factor as needed
                            color=colors[result['technique']])
            
            plt.legend(handles=legend_elements, title='Techniques',loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel('Runtime (seconds)')
            plt.ylabel('RMSE')
            plt.title(f'{self.ROI} Model Comparison')
            plt.gcf().patch.set_alpha(0)
            plt.show()