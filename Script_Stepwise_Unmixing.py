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


if __name__ == '__main__':
    print("Running Stepwise_Unmixing.py. Run from 'Run' notebook")
else:
    warnings.simplefilter(action='ignore', category=FutureWarning)
    class STEPWISE:
        def __init__(self, image_hdr_filename, image_filename, spectral_library_filename):
            self.load_new_image(image_hdr_filename, image_filename)
            self.load_new_spectral_library(spectral_library_filename)
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

        def forward_regression(self, y_index, threshold_in):
            if len(y_index) == 1:
                if y_index[0] < 0 or y_index[0] >= self.n_rows * self.n_cols:
                    raise ValueError("Invalid y_index. It should be between 0 and (n_rows * n_cols - 1).")
                self.y = self.reshape_image()[y_index, :].reshape(1, -1).T
            else:
                self.y = self.image_arr[y_index[0], y_index[1], :].flatten()

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
            if len(y_index) == 1:
                if y_index[0] < 0 or y_index[0] >= self.n_rows * self.n_cols:
                    raise ValueError("Invalid y_index. It should be between 0 and (n_rows * n_cols - 1).")
                self.y = self.reshape_image()[y_index, :].reshape(1, -1).T
            else:
                self.y = self.image_arr[y_index[0], y_index[1], :].flatten()

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

        def forward_selectedindex_fit(self, y_index=None):
            self.technique = "Forward Stepwise Regression"
            if len(y_index) == 1:
                if y_index[0] < 0 or y_index[0] >= self.n_rows * self.n_cols:
                    raise ValueError("Invalid y_index. It should be between 0 and (n_rows * n_cols - 1).")
                self.y = self.reshape_image()[y_index, :].reshape(1, -1).T
            else:
                self.y = self.image_arr[y_index[0], y_index[1], :].flatten()

            self.X = self.spectral_library.T
            self.modelselect = self.forward_regression(y_index, 0.05)
            self.model_X = self.X[:, list(self.modelselect)]

            model_coefficients, _ = nnls(self.model_X, self.y)
            self.summary_model_coefficients = model_coefficients
            self.plt_model_coefficients = model_coefficients


            '''forwardmodel = sm.OLS(self.y, model_X)
            forwardres = forwardmodel.fit()
            self.model_coefficients = forwardres.params'''

            
            non_zero_indices = [index for index, coefficient in zip(self.modelselect, model_coefficients) if coefficient != 0]
            self.non_zero_spectral_names = [self.spectra_names[index] for index in non_zero_indices]

            y_infer = np.dot(self.model_X, model_coefficients)
            self.rmse = np.sqrt(mean_squared_error(self.y, y_infer))
            
            
            #self.final_summary()
            #self.plot_spectra(modelselect,model_X)
            return self

        def backward_selectedindex_fit(self, y_index=None):
            self.technique = "Backward Stepwise Regression"
            if len(y_index) == 1:
                if y_index[0] < 0 or y_index[0] >= self.n_rows * self.n_cols:
                    raise ValueError("Invalid y_index. It should be between 0 and (n_rows * n_cols - 1).")
                self.y = self.reshape_image()[y_index, :].reshape(1, -1).T
            else:
                self.y = self.image_arr[y_index[0], y_index[1], :].flatten()

            self.X = self.spectral_library.T
            self.modelselect = self.backward_regression(y_index, 0.05)
            self.model_X = self.X[:, list(self.modelselect)]

            model_coefficients, _ = nnls(self.model_X, self.y)

            
            non_zero_indices = [index for index, coefficient in zip(self.modelselect, model_coefficients) if coefficient != 0]
            self.non_zero_spectral_names = [self.spectra_names[index] for index in non_zero_indices]
            self.plt_model_coefficients = model_coefficients
            #self.plot_spectra(self.modelselect,self.model_X)
            
            
            '''backwardmodel = sm.OLS(self.y, model_X)
            backwardres = backwardmodel.fit()
            self.model_coefficients = backwardres.params'''

            self.summary_model_coefficients = model_coefficients[np.where(model_coefficients != 0)[0]]

            y_infer = np.dot(self.model_X, model_coefficients)
            self.rmse = np.sqrt(mean_squared_error(self.y, y_infer))
                        
            #self.final_summary()

            return self
            
        
        def final_summary(self):
            summary = [(name, coefficient) for name, coefficient in zip(self.non_zero_spectral_names, self.summary_model_coefficients)]
            summary = sorted(summary, key=lambda x: x[1], reverse=True)

            print(f"Summary: {self.technique}")
            for name, coefficient in summary:
                print(f"{name}, Coefficient: {coefficient}")

            return summary

        def plot_spectra(self):
            plt.figure(figsize=(12, 7))
            plt.plot(self.wavelengths, self.y, label="Observed Spectrum",color="black")
            plt.plot(self.wavelengths, np.dot(self.plt_model_coefficients, self.model_X.T), label="Estimated Spectrum",color="red")
            
            for index, name in zip(self.modelselect, self.non_zero_spectral_names):
                spectrum = self.spectral_library.T[:, index]
                plt.plot(self.wavelengths, spectrum, label=f"{name}", linestyle='dotted')
            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance/Radiance')
            plt.title(f"{self.technique} \n RMSE: {self.rmse}")
            plt.legend()
            plt.show()
        
            
            

             

            
            


            
