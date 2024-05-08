import numpy as np 
import pandas as pd 
import spectral.io.envi as envi
from itertools import combinations
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from itertools import chain
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    print("Running BMA_Umixing_nnls.py. Run from 'Run' notebook")
else:
    
    class BMAquad:
        def __init__(self, image_hdr_filename, image_filename, spectral_library_filename):
            self.load_new_image(image_hdr_filename, image_filename)
            self.load_new_spectral_library(spectral_library_filename)
            self.reshape_image()
            

        def load_new_image(self, image_hdr_filename, image_filename):
            if image_hdr_filename.endswith('.hdr'):
                # Open the ENVI header and data files 
                self.image = envi.open(image_hdr_filename, image_filename)

                # Load image data 
                self.image_arr = self.image.load()  

                # Obtain and store dimensions of the hyperspectral image
                self.n_rows, self.n_cols, self.n_imbands = self.image_arr.shape

                #  Initialized or reset abundance maps attribute
                self.abundance_maps = None
            else:
                print("WARNING: library should be a .hdr file!")

        def load_new_spectral_library(self, spectral_library_filename):
            if spectral_library_filename.endswith('.hdr'):
                # Open the ENVI header file
                lib = envi.open(spectral_library_filename)

                # Wavelength values extracted from the bands of the library
                self.wavelengths = lib.bands.centers

                # The number of bands (wavelengths) in the spectral library
                self.n_bands = len(self.wavelengths)

                # The number of spectra (materials) in the spectral library
                self.n_spectra= len(lib.names)

                # The actual spectral data (reflectance or radiance values)
                self.spectral_library = lib.spectra

                # The names of the spectra
                self.spectra_names = lib.names  # Update spectra_names with material names
            else:
                print("WARNING: library should be a .hdr file!")
        
        def reshape_image(self):
            # Convert the 3D hyperspectral image array into a 2D array
            reshaped_image = self.image_arr.reshape(self.n_rows * self.n_cols, self.n_bands)
            return reshaped_image
              
        def selectedindex_fit(self, y_index=None, **kwargs):
            self.technique = "BMA with Linear and Quadratic Terms"
            if len(y_index) == 1:
                if y_index[0] < 0 or y_index[0] >= self.n_rows * self.n_cols:
                    raise ValueError("Invalid y_index. It should be between 0 and (n_rows * n_cols - 1).")
                self.y = self.reshape_image()[y_index, :].reshape(1, -1).T
            else:
                self.y = self.image_arr[y_index[0], y_index[1], :].flatten()

            self.X = self.spectral_library.T

            self.nRows, self.nCols = np.shape(self.X)
            self.likelihoods = np.zeros(self.nCols)
            self.coefficients = np.zeros(self.nCols)
            self.likelihoods_quad = np.zeros((self.nCols,self.nCols))
            self.coefficients_quad = np.zeros((self.nCols,self.nCols))
            self.probabilities = np.zeros(self.nCols)
            self.probabilities_quad = np.zeros((self.nCols,self.nCols))

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
                #print("iteration: ", self.num_elements)
                self.model_index_set = None
                iteration_max_likelihood = 0
                self.model_index_list = []
                self.model_likelihood = [] 
                
                for model_combination in current_model_set:
                    
                    model_X = self.X[:, list(model_combination)]
                    model_combination_quad = []
                    # incorporating the nonlinear terms
                    for i1 in range(self.num_elements):
                        for i2 in range(i1,self.num_elements):
                            idx1 = model_combination[i1]
                            idx2 = model_combination[i2]
                            #print(f'Appending nonlinear term {idx1}x{idx2}')
                            model_combination_quad.append([idx1,idx2])
                            model_append = np.reshape(np.multiply(self.X[:,idx1],self.X[:,idx2]), (self.n_bands,1))
                            model_X = np.hstack( (model_X, model_append))
                    
                    model_coefficients, _ = nnls(model_X, self.y)

                    # rss = sum((y-f(x))^2)
                    # MSE = 1/n * rss
                    #https://en.wikipedia.org/wiki/Bayesian_information_criterion
                    rss = np.sum((self.y - np.dot(model_X, model_coefficients)) ** 2)
                    k = model_X.shape[1]  
                    n = self.y.shape[0]
                    bic = n * np.log(rss / n) + k * np.log(n)
                    model_likelihood = np.exp(-bic / 2) * np.prod(self.Priors[list(model_combination)])
                    
                    if model_likelihood > iteration_max_likelihood:
                        iteration_max_likelihood = model_likelihood
                        self.model_index_set = model_combination
                        self.model_combination_quad = model_combination_quad
                        self.model_set_coefficients = model_coefficients
                                     
                    self.likelihood_sum += model_likelihood                  

                    self.model_index_list.append(model_combination)
                    self.model_likelihood.append(model_likelihood)
                    for i, model_idx in enumerate(model_combination):
                        self.likelihoods[model_idx] += model_likelihood
                        self.coefficients[model_idx] += model_coefficients[i] * model_likelihood
                    for i, model_idx in enumerate(model_combination_quad):
                        self.likelihoods_quad[model_idx[0],model_idx[1]] += model_likelihood
                        self.coefficients_quad[model_idx[0],model_idx[1]] += model_coefficients[self.num_elements + i] * model_likelihood
                        
                self.model_probability = np.asarray(self.model_likelihood)/self.likelihood_sum

                if iteration_max_likelihood > self.max_likelihood:
                    #############################
                    # NOTE FROM BILL: I am commenting these out. The BMA method should only be based on the average model, not a best model.
                    #############################
                    #self.best_avg_model_y = np.dot(self.X, self.coefficients/self.likelihood_sum)
                    #self.best_estimated_model_y = np.dot(self.X[:, list(self.model_index_set)], self.model_set_coefficients)                            
                    self.max_likelihood = iteration_max_likelihood
                    #self.best_model_index_name = [self.spectra_names[index] for index in self.model_index_set]
                    #self.best_model_index_set = self.model_index_set
                    #best_coefficients = model_coefficients
                    #self.best_likelihood_sum = self.likelihood_sum
                    #best_summary = self.summary()                                

                method1 = False
                if method1 == True:
                    top_models_threshold = round(0.05 * len(self.model_probability))
               
                    sorted_models = sorted(zip(self.model_index_list, self.model_probability), key=lambda x: x[1], reverse=True)
                    candidate_models = []
                    for i, (model_idx, model_prob) in enumerate(sorted_models):
                            if i < top_models_threshold:
                                for index in model_idx:
                                    if index not in candidate_models:
                                        candidate_models.append(index)
                    current_model_set = list(combinations(candidate_models, self.num_elements + 1)) 
                else:
                    top_models_threshold = round(0.05 * self.max_likelihood)

                    candidate_models = []
                    current_model_set = []
                    for i, (model_idx, model_likelihood) in enumerate(zip(self.model_index_list, self.model_likelihood)):
                            if model_likelihood > top_models_threshold:
                                for idx in range(self.nCols):
                                    current_model_set.append(model_idx + (idx,) if model_idx else (idx,))
                    #print(f"next_model_solution_space:{len(current_model_set)}")
                    
                    if len(current_model_set) == 0:
                        print("The number of variables required to for the next iteration exceed the number of candidate models")
                        print(f"BMA is finishing early at iteration: {self.num_elements}")
                        break                        
                
                #self.summary()
                #self.plot_spectra()                
                
                if top_models_threshold < self.num_elements + 1 or len(current_model_set) == 0:
                    print("The number of variables required to for the next iteration exceed the number of candidate models")
                    print(f"BMA is finishing early at iteration: {self.num_elements}")
                    break

            self.probabilities = self.likelihoods / self.likelihood_sum
            self.coefficients = self.coefficients / self.likelihood_sum   
            self.probabilities_quad = self.likelihoods_quad / self.likelihood_sum
            self.coefficients_quad = self.coefficients_quad / self.likelihood_sum   
            
            # coefficients and probabilities, just for individual material spectra
            self.probabilities_single_materials_only = self.probabilities/np.sum(self.probabilities)
            self.coefficients_single_materials_only = self.coefficients/np.sum(self.probabilities)

            # This is the average model that should be used from model averaging
            #y_infer = np.dot(self.X, self.coefficients) #(This computation is fine for linear models)
            self.y_infer = np.zeros(self.nRows)
            for i in range(self.nCols):
                self.y_infer +=  self.X[:,i]*self.coefficients[i] # contribution from linear terms
                for j in range(self.nCols):
                    self.y_infer +=  np.multiply(self.X[:,i],self.X[:,j])*self.coefficients_quad[i,j] # contribution from quadratic terms
                    
            
            self.rmse = np.sqrt(mean_squared_error(self.y, self.y_infer))
            
            # select high probability spectra to plot with the model
            # selecting spectra with a probbility greater than 0.01
            self.model_index_set = []
            self.model_index_name = []
            spectra_indices_sorted_by_probablity = np.argsort(-self.probabilities_single_materials_only)
            for i in spectra_indices_sorted_by_probablity:
                if self.probabilities_single_materials_only[i] > 0.01:
                    self.model_index_set.append(i)
                    self.model_index_name.append(self.spectra_names[i]+', p='+str(self.probabilities_single_materials_only[i]))
            

            #print("BMA Best Model:", self.best_model_index_set,"|",self.max_likelihood)

            #self.plot_spectra()
            #self.final_summary()

            return self
        def plot_spectra(self):
                plt.figure(figsize=(20, 8))
                plt.title(f"{self.technique}\nRMSE = {self.rmse} ")
                avg_model_y = self.y_infer
                #if self.model_index_set == self.best_model_index_set:
                #    avg_model_y = self.best_avg_model_y
                #    estimated_model_y = self.best_estimated_model_y
                #else:
                #    avg_model_y = np.dot(self.X, self.coefficients/self.likelihood_sum)
                #    estimated_model_y = np.dot(self.X[:, list(self.model_index_set)], self.model_set_coefficients)
                plt.plot(self.wavelengths,avg_model_y, label='Avg Model', color = "black",linewidth = 4, alpha = 0.2)
                plt.plot(self.wavelengths, self.y.flatten(), label='Observed Spectrum', linestyle='dashed')
                #plt.plot(self.wavelengths, estimated_model_y, label='Estimated Model', linestyle='dotted')
                for idx,name in zip(self.model_index_set,self.model_index_name):
                    plt.plot(self.wavelengths, self.spectral_library[idx], label=name)
                plt.xlabel('Wavelength')
                plt.ylabel('Reflectance/Radiance')
                plt.legend()
                plt.show()
        
        def summary(self,):
            probabilities_temp = self.likelihoods / self.likelihood_sum
            coefficients_temp = self.coefficients / self.likelihood_sum
            
            if self.model_index_set is not None:
                best_model_spectral_names = [self.spectra_names[i] for i in self.model_index_set ] 
                best_model_probabilities = [probabilities_temp[i] for i in self.model_index_set ]
                best_model_coefficients = [coefficients_temp[i] for i in self.model_index_set ]

                df = pd.DataFrame([best_model_spectral_names, best_model_probabilities, best_model_coefficients, self.model_set_coefficients], 
                    ["Spectra", "Prob", "Avg.Coeff", "Best Model Coeff"]).T
                '''with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
                    print(df)'''
            else:
                print("No best model index set.")
                
            return df                   
        
        def final_summary(self):
            df = pd.DataFrame([self.spectra_names, self.probabilities, self.coefficients], 
                    ["Variable Name", "Probability", "Coefficient"]).T
            df.sort_values("Probability",inplace=True, ascending = False)
            print(f"Summary: {self.technique}")
            #print(df.head(10))
            with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
                print(df)
            
        def final_summary_compare(self):
            summary = [(name, coefficient) for name, coefficient in zip(self.spectra_names,self.coefficients)]
            summary = sorted(summary, key=lambda x: x[1], reverse=True)

            '''print(f"Summary: {self.technique}")
            for name, coefficient in summary:
                print(f"{name}, Coefficient: {coefficient}")'''

            return summary
            

             

            
            


            
