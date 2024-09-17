import importlib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from IPython.display import display  
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy.optimize import nnls
from pyomo.environ import *
from pyomo.opt import SolverFactory
import re
from matplotlib.lines import Line2D
import time
import csv
from collections import Counter
from pyomo.opt import SolverFactory
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    print("Running LASSO_Unmixing.py. Run from 'Run' notebook")
else:
    class unmix:
        def __init__(self, image_hdr_filename, image_filename, spectral_library_filename,region_of_interest,ROI_size):
            self.load_new_image(image_hdr_filename, image_filename)
            self.load_new_spectral_library(spectral_library_filename)
            self.reshape_image()
            self.load_index_to_unmix(region_of_interest, ROI_size)
            self.load_chemical_data()
            self.aceDetect(self.image_arr)



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
                self.nLibSpec = len(self.spectra_names)

            else:
                print("WARNING: library should be a .hdr file!")

        def reshape_image(self):
            reshaped_image = self.image_arr.reshape(self.n_rows * self.n_cols, self.n_bands)
            return reshaped_image
        
        def load_index_to_unmix(self,region_of_interest,ROI_size): 
            self.index_samples = []
            if region_of_interest == 1:                
                self.ROI_name = "alunite"
                self.df = pd.read_csv("alunite_hills_ROIs.csv", encoding='latin1')
                if ROI_size == "s":
                    self.index_samples=[[305,90]]
                elif ROI_size == "m":
                    self.index_samples=[[304,88],[304,89],[304,90],[305,88],[305,89],[305,90],[305,91],[306,88],[306,89],[306,90]]
                else:                       
                    df_index_samples = self.df.copy()  
                    df_index_samples.columns = df_index_samples.columns
                    for index, row in df_index_samples.iterrows():
                        pixel_x = row[" Pixel_x"]
                        pixel_y = row[" Pixel_y"]
                        self.index_samples.append([pixel_y, pixel_x])
            elif region_of_interest == 2:
                self.ROI_name = "kaolinite"
                self.df = pd.read_csv("kaolinite_hills_ROIs.csv", encoding='latin1')
                if ROI_size == "s":
                    self.index_samples=[[290,290]]
                elif ROI_size == "m":
                    self.index_samples=[[289,288],[289,289],[289,290],[290,288],[290,289],[290,290],[290,291],[291,288],[291,289],[291,290]]
                else:                       
                    df_index_samples = self.df.copy()  
                    df_index_samples.columns = df_index_samples.columns
                    for index, row in df_index_samples.iterrows():
                        pixel_x = row[" Pixel_x"]
                        pixel_y = row[" Pixel_y"]
                        self.index_samples.append([pixel_y, pixel_x])
            else:
                self.ROI_name = "montmorillonite"
                self.df = pd.read_csv("montmorillonite_hills_ROIs.csv", encoding='latin1')
                if ROI_size == "s":
                    self.index_samples=[[349,48]]
                elif ROI_size == "m":
                    self.index_samples=[[347,47],[347,48],[347,49],[348,47],[348,48],[348,49],[349,47],[349,48],[349,49],[349,50]]
                else:                     
                    df_index_samples = self.df.copy()  
                    df_index_samples.columns = df_index_samples.columns
                    for index, row in df_index_samples.iterrows():
                        pixel_x = row[" Pixel_x"]
                        pixel_y = row[" Pixel_y"]
                        self.index_samples.append([pixel_y, pixel_x])
                                
        def load_chemical_data(self):
            self.chemicaldf = pd.read_csv("mineraldata2.csv", encoding='latin1')         
            caterogy_chem = self.chemicaldf.iloc[:, 1].tolist()
            caterogy_chem = pd.Series(caterogy_chem).unique()
            caterogy_chem = pd.Series(caterogy_chem)
            caterogy_chem = caterogy_chem[:16]
            self.mineral_categories  = caterogy_chem.tolist()            


        #########LASSO#############
        
        def lasso_unmix(self, alpha_min=0.0001, alpha_max=0.1, step_size=0.001, cv=5):
            print(f'Target Mineral: {self.ROI_name}')
            self.technique = "LASSO Regression"
            print(self.technique)

            #initialize objects
            lasso_acc_count =0
            model_size_list_lasso = []
            RMSE_list_lasso = []
            run_time_list_lasso = []
            self.lasso_chem_cat = []
            self.lasso_ordered_names = []  # New list to store ordered names
            self.lasso_ordered_coefficients = []
            lasso_spec = []
            precision_lasso = []
            alphas = np.arange(alpha_min, alpha_max + step_size, step_size)
                      
            for y_index in self.index_samples:

                start_time = time.time()          

                self.y = self.image_arr[y_index[0], y_index[1], :].flatten()
                self.X = self.spectral_library.T

                lasso_model = LassoCV(alphas=alphas, cv=cv)
                lasso_model.fit(self.X, self.y)
                self.model_coefficients = lasso_model.coef_

                self.non_zero_indices = np.where(self.model_coefficients != 0)[0]
                self.non_zero_coefficients = self.model_coefficients[self.non_zero_indices]
                self.non_zero_spectral_names = [self.spectra_names[index] for index in self.non_zero_indices]

                # Order the minerals by their coefficients
                sorted_minerals = sorted(
                    zip(self.non_zero_spectral_names, self.non_zero_coefficients), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                self.ordered_names, self.ordered_coefficients = zip(*sorted_minerals)
                self.lasso_ordered_names.append(self.ordered_names)  # Store ordered names
                self.lasso_ordered_coefficients.append(self.ordered_coefficients)
                
                y_infer = np.dot(self.X[:, self.model_coefficients != 0], self.non_zero_coefficients)
                
                end_time = time.time()  # End the timer
                elapsed_time = end_time - start_time 

                self.lasso_mineral_categories = [] 
                for name in self.ordered_names:                
                    cat = self.chemicaldf[self.chemicaldf['Name']==name.split()[1]].iloc[0:1].iloc[0]['Category']
                    self.lasso_mineral_categories.append(cat)

                
                rmse = np.sqrt(mean_squared_error(self.y, y_infer))
                model_size = len(self.ordered_names)

                for name, coefficient, category in zip(self.ordered_names, self.ordered_coefficients, self.lasso_mineral_categories):
                    print(f"{name}, Coefficient: {coefficient}, Category: {category}")
                
                RMSE_list_lasso.append(rmse)
                model_size_list_lasso.append(model_size)
                run_time_list_lasso.append(elapsed_time)
                self.lasso_chem_cat.append(self.lasso_mineral_categories)
                lasso_spec.append(self.ordered_names)
                precision_at_k = self.precison_at_k_calc(round(model_size*.5))
                print(f"Precision at k:{precision_at_k}")
                precision_lasso.append(precision_at_k)
                target_detect = self.count_mineral_occurrence(self.ordered_names)
                if target_detect>0:
                    lasso_acc_count = lasso_acc_count + 1
                print("------------------------------------------")
            # Calculate averages
            self.num_samples = len(self.index_samples)
            self.avg_RMSE_lasso = sum(RMSE_list_lasso) / self.num_samples
            self.avg_run_time_lasso = sum(run_time_list_lasso) / self.num_samples
            self.avg_model_size_lasso = round(sum(model_size_list_lasso) / self.num_samples)
            self.avg_precision_lasso = sum(precision_lasso) / self.num_samples
            self.lasso_category_counts = self.count_categories(self.lasso_chem_cat, self.lasso_mineral_categories)
            self.lasso_mineral_count = self.count_mineral_occurrences(lasso_spec)
            self.lasso_percent = lasso_acc_count/self.num_samples

            print(f"Average RMSE: {self.avg_RMSE_lasso}")
            print(f"Average Run Time: {self.avg_run_time_lasso}")
            print(f"Average Model Size: {self.avg_model_size_lasso}")
            print(f"Average Precision at k: {self.avg_precision_lasso}")
            print(f"Target Mineral Detection Percentage: {self.lasso_percent * 100}%")

            self.export_data_to_csv()
            return self
        
#########STEP_WISE#############
        
        def forward_regression(self, y_index, threshold_in):

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
        
        def forward_unmix(self):
            print(f'Target Mineral: {self.ROI_name}')

            self.technique = "Forward Stepwise Regression"
            print(self.technique)

            #initialize objects
            forward_acc_count =0
            model_size_list_forward = []
            RMSE_list_forward = []
            run_time_list_forward = []
            self.forward_chem_cat = []
            self.forward_ordered_names = []  # New list to store ordered names
            self.forward_ordered_coefficients = []
            forward_spec = []
            precision_forward = []

            for y_index in self.index_samples:

                start_time = time.time()

                self.y = self.image_arr[y_index[0], y_index[1], :].flatten()

                self.X = self.spectral_library.T
                self.modelselect = self.forward_regression(y_index, 0.05)
                self.model_X = self.X[:, list(self.modelselect)]

                model_coefficients, _ = nnls(self.model_X, self.y)
                self.summary_model_coefficients = model_coefficients
                self.plt_model_coefficients = model_coefficients

                self.non_zero_indices = [index for index, coefficient in zip(self.modelselect, model_coefficients) if coefficient != 0]
                self.non_zero_spectral_names = [self.spectra_names[index] for index in self.non_zero_indices]
                
                # Order the minerals by their coefficients
                sorted_minerals = sorted(
                    zip(self.non_zero_spectral_names, self.plt_model_coefficients), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                self.ordered_names, self.ordered_coefficients = zip(*sorted_minerals)
                self.forward_ordered_names.append(self.ordered_names)
                self.forward_ordered_coefficients.append(self.ordered_coefficients)
                
                y_infer = np.dot(self.model_X, model_coefficients)

                end_time = time.time()  # End the timer
                elapsed_time = end_time - start_time
                
                self.forward_mineral_categories = [] 
                for name in self.ordered_names:                
                    cat = self.chemicaldf[self.chemicaldf['Name']==name.split()[1]].iloc[0:1].iloc[0]['Category']
                    self.forward_mineral_categories.append(cat)

                rmse = np.sqrt(mean_squared_error(self.y, y_infer))  
                model_size = len(self.ordered_names)   

                for name, coefficient, category in zip(self.ordered_names, self.ordered_coefficients, self.forward_mineral_categories):
                    print(f"{name}, Coefficient: {coefficient}, Category: {category}") 


                RMSE_list_forward.append(rmse)
                model_size_list_forward.append(model_size)
                run_time_list_forward.append(elapsed_time)
                self.forward_chem_cat.append(self.forward_mineral_categories)
                forward_spec.append(self.ordered_names)
                precision_at_k = self.precison_at_k_calc(round(model_size*.5))
                print(f"Precision at k:{precision_at_k}")
                precision_forward.append(precision_at_k)
                target_detect = self.count_mineral_occurrence(self.ordered_names)
                if target_detect>0:
                    forward_acc_count = forward_acc_count + 1
                print("------------------------------------------")
            # Calculate averages
            self.num_samples = len(self.index_samples)
            self.avg_RMSE_forward = sum(RMSE_list_forward) / self.num_samples
            self.avg_run_time_forward = sum(run_time_list_forward) / self.num_samples
            self.avg_model_size_forward = round(sum(model_size_list_forward) / self.num_samples)
            self.avg_precision_forward = sum(precision_forward)/self.num_samples
            self.forward_category_count = self.count_categories(self.forward_chem_cat, self.forward_mineral_categories)
            self.forward_mineral_count = self.count_mineral_occurrences(forward_spec)
            self.forward_percent = forward_acc_count/self.num_samples   

            print(f"Average RMSE: {self.avg_RMSE_forward}")
            print(f"Average Run Time: {self.avg_run_time_forward}")
            print(f"Average Model Size: {self.avg_model_size_forward}")
            print(f"Average Precision at k: {self.avg_precision_forward}")
            print(f"Target Mineral Detection Percentage: {self.forward_percent * 100}%")    

            self.export_data_to_csv()          
            return self
        
#########Mixed_Integer_Non-linear_Program#############
        def MINLP_unmix(self):
            print(f'Target Mineral: {self.ROI_name}')

            self.technique = "Mixed_Integer_Non-linear_Program"
            print(self.technique)     

            #initialize objects
            MINLP_acc_count =0
            no_model_count = 0
            model_size_list_MINLP = []
            RMSE_list_MINLP = []
            run_time_list_MINLP = []
            self.MINLP_chem_cat = []
            self.MINLP_ordered_names = []  # New list to store ordered names
            self.MINLP_ordered_coefficients = []
            MINLP_spec = []
            precision_MINLP = []

            for y_index in self.index_samples:

                start_time = time.time() 

                # Initiate Model
                model = ConcreteModel()

                # Data
                spectra_data = {i: self.spectral_library[i, :].tolist() for i in range(self.spectral_library.shape[0])}
                spectra_indices = list(spectra_data.keys())

                # Sets
                model.I = Set(initialize=spectra_indices)
                model.W = Set(initialize=range(50))

                # Parameters
                model.B = Param(model.I, initialize={i: 2 for i in spectra_indices})
                model.y = self.image_arr[y_index[0], y_index[1], :].flatten()

                # Decision Variables
                model.x = Var(model.I, domain=Binary)
                model.a = Var(model.I, domain=NonNegativeReals)

                # Objective Function
                def obj_rule(model):
                    return sum((model.y[j] - sum(model.a[i] * spectra_data[i][j] for i in model.I))**2 for j in model.W)
                model.obj = Objective(rule=obj_rule, sense=minimize)

                # Constraints
                def spectra_constraint_rule(model, i):
                    return model.B[i] * model.x[i] >= model.a[i]
                model.spectra_constraint = Constraint(model.I, rule=spectra_constraint_rule)
                        
                def modelsize_rule(model):
                    return sum(model.x[i] for i in model.I) <= 4
                model.modelsize_rule_constraint = Constraint(rule=modelsize_rule)

                # Solve
                solver = SolverFactory('mindtpy')
                results = solver.solve(model,time_limit=300)

                end_time = time.time()  # End the timer
                elapsed_time = end_time - start_time

                self.non_zero_spectral_names = []
                self.non_zero_indices = []
                self.non_zero_coefficients = []

                # Check solver status and print the results
                if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
                    rmse = np.sqrt(value(model.obj) / 50)
                    RMSE_list_MINLP.append(rmse)
                    for i in model.I:
                        if model.x[i].value > 0:
                            #print(f"x[{i}] {self.spectra_names[i]} = {model.x[i].value}, a[{i}] = {model.a[i].value}")
                            self.non_zero_spectral_names.append(self.spectra_names[i])
                            model_size = len(self.non_zero_spectral_names)  
                            self.non_zero_indices.append(i)
                            self.non_zero_coefficients.append(model.a[i].value)

                    # Order the minerals by their coefficients
                    sorted_minerals = sorted(
                        zip(self.non_zero_spectral_names, self.non_zero_coefficients), 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )
                    self.ordered_names, self.ordered_coefficients = zip(*sorted_minerals)
                    self.MINLP_ordered_names.append(self.ordered_names)
                    self.MINLP_ordered_coefficients.append(self.ordered_coefficients)

                    self.MINLP_mineral_categories = [] 
                    for name in self.ordered_names:                
                        cat = self.chemicaldf[self.chemicaldf['Name']==name.split()[1]].iloc[0:1].iloc[0]['Category']
                        self.MINLP_mineral_categories.append(cat)                
                    
                    for name, coefficient, category in zip(self.ordered_names, self.ordered_coefficients, self.MINLP_mineral_categories):
                        print(f"{name}, Coefficient: {coefficient}, Category: {category}")
                    
                    model_size_list_MINLP.append(model_size)
                    run_time_list_MINLP.append(elapsed_time)
                    self.MINLP_chem_cat.append(self.MINLP_mineral_categories)
                    MINLP_spec.append(self.ordered_names)
                    precision_at_k = self.precison_at_k_calc(round(model_size*.5))
                    print(f"Precision at k:{precision_at_k}")
                    precision_MINLP.append(precision_at_k)
                    target_detect = self.count_mineral_occurrence(self.ordered_names)
                    if target_detect>0:
                        MINLP_acc_count = MINLP_acc_count + 1
                    print("------------------------------------------")

                else:
                    no_model_count = no_model_count + 1
                    print("No optimal solution found.")
                    continue
                
                   
            # Calculate averages
            self.num_samples = len(self.index_samples) - no_model_count
            self.avg_RMSE_MINLP = sum(RMSE_list_MINLP) / self.num_samples
            self.avg_run_time_MINLP = sum(run_time_list_MINLP) / self.num_samples
            self.avg_model_size_MINLP = round(sum(model_size_list_MINLP) / self.num_samples)
            self.avg_precision_MINLP = sum(precision_MINLP) / self.num_samples
            self.MINLP_category_counts = self.count_categories(self.MINLP_chem_cat, self.MINLP_mineral_categories)
            self.MINLP_target_count = self.count_mineral_occurrences(MINLP_spec)
            self.MINLP_percent = MINLP_acc_count/self.num_samples

            print(f"Average RMSE: {self.avg_RMSE_MINLP}")
            print(f"Average Run Time: {self.avg_run_time_MINLP}")
            print(f"Average Model Size: {self.avg_model_size_MINLP}")
            print(f"Average Precision at k: {self.avg_precision_MINLP}")
            print(f"Target Mineral Detection Percentage: {self.MINLP_percent * 100}%")

            self.export_data_to_csv()
            return self
            

#########HySudeB#############
        def HySudeB_unmix(self, alpha_min=0.001, alpha_max=0.9, step_size=0.001, cv=5):
            print(f'Target Mineral: {self.ROI_name}')

            self.technique = "HySudeB"
            print(self.technique)

            #initialize objects
            HySudeB_acc_count = 0
            model_size_list_HySudeB = []
            RMSE_list_HySudeB = []
            run_time_list_HySudeB = []
            self.HySudeB_chem_cat = []
            self.HySudeB_ordered_names = []  # New list to store ordered names
            self.HySudeB_ordered_coefficients = []
            HySudeB_spec = []
            precision_HySudeB = []
            alphas = np.arange(alpha_min, alpha_max + step_size, step_size)   
            
            #Load the spectra for the ROI from the image,
            #using pixel coordinates from the ROI csv file
            nPix,hypsec_col = self.df.shape
            ROI_spectra = np.zeros((nPix,self.n_imbands))
            for i,row in self.df.iterrows():
                pass
                x_coord = row[' Pixel_x']
                y_coord = row[' Pixel_y']
                ROI_spectra[i,:] = self.image_arr[x_coord,y_coord,:].flatten()
            
            # compute the statistics
            cov_type = 'Im'
            if cov_type=='ROI':
                m = np.mean(ROI_spectra, axis=0)
                c = np.cov(ROI_spectra.T)
            else:
                im_list = np.reshape(self.image_arr, (self.n_rows*self.n_cols, self.n_imbands))
                m = np.mean(im_list, axis=0)
                c = np.cov(im_list.T)

            evals,evecs = np.linalg.eig(c)
            # truncate the small eigenvalues to stablize the invers
            evals[evals<10**(-4)] = 10**(-4)

            D = np.diag(evals**(-1/2)) # this is the square root of D^(-1) from the paper
            W = np.matmul(evecs,D)
            W.shape

            for y_index in self.index_samples:
                start_time = time.time() 
                # read the pixel
                self.y = self.image_arr[y_index[0], y_index[1], :].flatten()
                #y = self.image_arr[x_coord,y_coord,:].flatten()
                self.X = self.spectral_library.T

                # Whiten the pixel spectrum
                yW = np.matmul(W.T,(self.y-m))

                # Step 1: Subtract the mean
                X_meansub = np.zeros((self.n_imbands,self.nLibSpec))
                # Subtract the mean from each spectra in the library
                for i in range(self.n_imbands):
                    X_meansub[i,:] = self.X[i,:]-m[i]

                # Step 2: multiply by W.T
                XW = np.matmul(W.T,X_meansub)

                lasso_model = LassoCV(alphas=alphas, cv=cv, positive=True) # NOTE: change alpha for raw data, image, or ROI
                lasso_model.fit(XW, yW)
                self.model_coefficients = lasso_model.coef_

                self.non_zero_indices = np.where(self.model_coefficients != 0)[0]
                self.non_zero_coefficients = self.model_coefficients[self.non_zero_indices]
                self.non_zero_spectral_names = [self.spectra_names[index] for index in self.non_zero_indices]

                y_infer = np.dot(self.X[:, self.model_coefficients != 0], self.non_zero_coefficients)

                end_time = time.time()  # End the timer
                elapsed_time = end_time - start_time


                if self.model_coefficients.max() == 0.0:
                    continue
                else:
                    
                    # Order the minerals by their coefficients
                    sorted_minerals = sorted(
                        zip(self.non_zero_spectral_names, self.non_zero_coefficients), 
                        key=lambda x: abs(x[1]), 
                        reverse=True
                    )
                    self.ordered_names, self.ordered_coefficients = zip(*sorted_minerals)
                    self.HySudeB_ordered_names.append(self.ordered_names)  # New list to store ordered names
                    self.HySudeB_ordered_coefficients.append(self.ordered_coefficients)    

                    self.HySudeB_mineral_categories = [] 
                    for name in self.ordered_names:                
                        cat = self.chemicaldf[self.chemicaldf['Name']==name.split()[1]].iloc[0:1].iloc[0]['Category']
                        self.HySudeB_mineral_categories.append(cat)              
                                    
                    rmse = np.sqrt(mean_squared_error(self.y, y_infer))
                    model_size = len(self.ordered_names)  
                    
                    for name, coefficient, category in zip(self.ordered_names, self.ordered_coefficients, self.HySudeB_mineral_categories):
                        print(f"{name}, Coefficient: {coefficient}, Category: {category}")
                    
                    RMSE_list_HySudeB.append(rmse)
                    model_size_list_HySudeB.append(model_size)
                    run_time_list_HySudeB.append(elapsed_time)
                    self.HySudeB_chem_cat.append(self.HySudeB_mineral_categories)
                    HySudeB_spec.append(self.ordered_names)
                    precision_at_k = self.precison_at_k_calc(round(model_size*.5))
                    print(f"Precision at k:{precision_at_k}")
                    precision_HySudeB.append(precision_at_k)
                    target_detect = self.count_mineral_occurrence(self.ordered_names)
                    if target_detect>0:
                        HySudeB_acc_count = HySudeB_acc_count + 1
                    print("------------------------------------------")
            # Calculate averages
            self.num_samples = len(self.index_samples)
            self.avg_RMSE_HySudeB = sum(RMSE_list_HySudeB) / self.num_samples
            self.avg_run_time_HySudeB = sum(run_time_list_HySudeB) / self.num_samples
            self.avg_model_size_HySudeB = round(sum(model_size_list_HySudeB) / self.num_samples)
            self.avg_precision_HySudeB = sum(precision_HySudeB) / self.num_samples
            self.HySudeB_category_counts = self.count_categories(self.HySudeB_chem_cat, self.HySudeB_mineral_categories)
            self.HySudeB_target_count = self.count_mineral_occurrences(HySudeB_spec)
            self.HySudeB_percent = HySudeB_acc_count/self.num_samples

            print(f"Average RMSE: {self.avg_RMSE_HySudeB}")
            print(f"Average Run Time: {self.avg_run_time_HySudeB}")
            print(f"Average Model Size: {self.avg_model_size_HySudeB}")
            print(f"Average Precision at k: {self.avg_precision_HySudeB}")
            print(f"Target Mineral Detection Percentage: {self.HySudeB_percent * 100}%")

            self.export_data_to_csv()
            return self
        
        def count_categories(self, category_list, categories):
            flat_list = [item for sublist in category_list for item in sublist]
            counts = Counter(flat_list)
            return {cat: counts.get(cat, 0) for cat in categories}
        
        def count_mineral_occurrence(self,spec_list):
            return sum(self.ROI_name in spec.lower() for spec in spec_list)

        def count_mineral_occurrences(self,spec_list):
            return sum(self.ROI_name in spec for sublist in spec_list for spec in sublist)
        
        def precison_at_k_calc(self, k):
            if k == 0 or k > len(self.ordered_names):
                return 0.0  # Edge cases where k is zero or exceeds the length of the list

            roi_name_lower = self.ROI_name.lower()
            roi_indices = [i for i, name in enumerate(self.ordered_names) if roi_name_lower in name.lower()]

            # Calculate the number of relevant items in the top-k elements
            relevant_count = sum(1 for i in roi_indices if i < k)

            # Precision at k is the fraction of relevant items in the top-k
            precision_k = relevant_count / k
            return precision_k
        
        def precision_plot(self):
            techniques = ['Lasso', 'Forward Selection', 'MINLP', 'HySudeB']
            precisions = [self.avg_precision_lasso, self.avg_precision_forward, self.avg_precision_MINLP, self.avg_precision_HySudeB]

            # Combine and sort the data
            sorted_indices = sorted(range(len(precisions)), key=lambda i: precisions[i], reverse=True)
            sorted_techniques = [techniques[i] for i in sorted_indices]
            sorted_precisions = [precisions[i] for i in sorted_indices]

            # Plot the vertical bar chart
            plt.figure(figsize=(13, 6))
            bars = plt.barh(sorted_techniques, sorted_precisions, color='blue')  # Single color for all bars
            plt.title(f'{self.ROI_name} Detection')
            plt.xlabel('Average Precision')

            # Add annotations (numbers) on the bars with floating-point format
            for bar in bars:
                plt.text(bar.get_width() + 0.01, 
                        bar.get_y() + bar.get_height() / 2, 
                        f'{bar.get_width():.2f}',  # Display precision values with 2 decimal places
                        ha='left', va='center')

            plt.show()

        def comp_results(self):
            # Data for the scatter plot
            techniques = ['Lasso', 'Forward', 'MINLP', 'HySudeB']
            avg_RMSE = [self.avg_RMSE_lasso, self.avg_RMSE_forward, self.avg_RMSE_MINLP, self.avg_RMSE_HySudeB]
            avg_run_time = [self.avg_run_time_lasso, self.avg_run_time_forward, self.avg_run_time_MINLP, self.avg_run_time_HySudeB]
            avg_model_size = [self.avg_model_size_lasso, self.avg_model_size_forward, self.avg_model_size_MINLP, self.avg_model_size_HySudeB]

            # Create scatter plot
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(avg_RMSE, avg_run_time, s=[size*100 for size in avg_model_size], alpha=0.7)

            # Add labels for each point
            for i, technique in enumerate(techniques):
                plt.text(avg_RMSE[i], avg_run_time[i], technique, fontsize=12, ha='left')

            # Add title and labels
            plt.title('Comparison of Unmixing Techniques')
            plt.xlabel('Average RMSE')
            plt.ylabel('Average Computation Time (s)')
            plt.grid(True)

        def plot_mineral_categories(self):
            # Function to ensure all categories are present
            def ensure_all_categories(count_dict, categories):
                return {cat: count_dict.get(cat, 0) for cat in categories}

            # Function to sort the categories by count
            def sort_counts(count_dict):
                return dict(sorted(count_dict.items(), key=lambda item: item[1], reverse=False))

            # Function to add annotations on bars
            def add_annotations(ax, counts):
                for i, (category, count) in enumerate(counts.items()):
                    ax.text(count + 0.1, i, str(count), ha='left', va='center')

            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(25, 12))

            # Sort and plot Lasso Histogram
            sorted_lasso_counts = sort_counts(ensure_all_categories(self.lasso_category_counts, self.lasso_mineral_categories))
            axes[0, 0].barh(list(sorted_lasso_counts.keys()), list(sorted_lasso_counts.values()), color='blue')
            axes[0, 0].set_title('Lasso Mineral Categories')
            add_annotations(axes[0, 0], sorted_lasso_counts)

            # Sort and plot Forward Selection Histogram
            sorted_forward_counts = sort_counts(ensure_all_categories(self.forward_category_count, self.forward_mineral_categories))
            axes[0, 1].barh(list(sorted_forward_counts.keys()), list(sorted_forward_counts.values()), color='green')
            axes[0, 1].set_title('Forward Selection Mineral Categories')
            add_annotations(axes[0, 1], sorted_forward_counts)

            # Sort and plot MINLP Histogram
            sorted_MINLP_counts = sort_counts(ensure_all_categories(self.MINLP_category_counts, self.MINLP_mineral_categories))
            axes[1, 0].barh(list(sorted_MINLP_counts.keys()), list(sorted_MINLP_counts.values()), color='red')
            axes[1, 0].set_title('MINLP Mineral Categories')
            add_annotations(axes[1, 0], sorted_MINLP_counts)

            # Sort and plot HySudeB Histogram
            sorted_HySudeB_counts = sort_counts(ensure_all_categories(self.HySudeB_category_counts, self.HySudeB_mineral_categories))
            axes[1, 1].barh(list(sorted_HySudeB_counts.keys()), list(sorted_HySudeB_counts.values()), color='purple')
            axes[1, 1].set_title('HySudeB Mineral Categories')
            add_annotations(axes[1, 1], sorted_HySudeB_counts)

            # Adjust the layout to provide more space between plots
            plt.subplots_adjust(wspace=0.3, hspace=0.4)

            plt.show()

        def plot_overall_minerals(self):
            # Function to count occurrences
            def count_categories(category_list, categories):
                flat_list = [item for sublist in category_list for item in sublist]
                counts = Counter(flat_list)
                return {cat: counts.get(cat, 0) for cat in categories}

            # Ensure all categories are counted, even if they are zero
            def ensure_all_categories(count_dict, categories):
                return {cat: count_dict.get(cat, 0) for cat in categories}

            # Count occurrences for each technique
            lasso_counts = ensure_all_categories(count_categories(self.lasso_chem_cat, self.lasso_mineral_categories), self.lasso_mineral_categories)
            forward_counts = ensure_all_categories(count_categories(self.forward_chem_cat, self.forward_mineral_categories), self.forward_mineral_categories)
            MINLP_counts = ensure_all_categories(count_categories(self.MINLP_chem_cat, self.MINLP_mineral_categories), self.MINLP_mineral_categories)
            HySudeB_counts = ensure_all_categories(count_categories(self.HySudeB_chem_cat, self.HySudeB_mineral_categories), self.HySudeB_mineral_categories)

            # Combine all counts for the overall histogram
            combined_counts = Counter(lasso_counts) + Counter(forward_counts) + Counter(MINLP_counts) + Counter(HySudeB_counts)

            # Sort the combined counts
            sorted_categories = sorted(combined_counts.keys(), key=lambda x: combined_counts[x], reverse=True)
            sorted_counts = [combined_counts[cat] for cat in sorted_categories]

            # Plot the horizontal bar chart
            plt.figure(figsize=(12, 8))
            bars = plt.barh(sorted_categories, sorted_counts, color='gray')
            plt.title('Overall Mineral Categories')
            plt.xlabel('Count')

            # Add annotations (numbers) on the bars
            for bar in bars:
                plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, int(bar.get_width()), ha='left', va='center')

            plt.show()

        def aceDetect(self, im_arr):
            # Select the spectra associated with the Region of Interest (ROI)
            spectra_indices = [i for i, name in enumerate(self.spectra_names) if self.ROI_name.lower() in name.lower()]

            if len(spectra_indices) == 0:
                print(f"Target spectrum for {self.ROI_name} not found in the library.")
                return None
            elif len(spectra_indices) > 1:
                print(f"Multiple spectra found for {self.ROI_name}. Taking the mean.")
                t = np.mean([self.spectral_library[i] for i in spectra_indices], axis=0)
            else:
                print(f"Single spectrum found for {self.ROI_name}.")
                t = self.spectral_library[spectra_indices[0]]

            # Image dimensions and pixel count
            nRows, nCols, nBands = im_arr.shape
            nPix = nRows * nCols

            # Reshape image for processing
            im_list = im_arr.reshape(nPix, nBands)

            # Compute the mean of the image spectra
            mu = np.mean(im_list, axis=0)

            # Compute the covariance matrix of the image spectra
            C = np.cov(im_list.T)

            # Eigenvalue decomposition
            evals, evecs = np.linalg.eig(C)

            # Whitening transformation
            DiagMatrix = np.diag(evals**(-1/2))
            W = np.matmul(evecs, DiagMatrix)

            # Normalize the target spectrum
            t_W = np.matmul(W.T, (t - mu))

            # Normalize the image (demean)
            im_demean = im_list - mu
            im_W = np.matmul(W.T, im_demean.T).T

            # Compute the ACE correlation for each pixel
            denom = np.sqrt(np.sum(im_W**2, axis=1)) * np.sqrt(np.sum(t_W**2))
            numerator = np.dot(im_W, t_W)
            D = numerator / denom

            # Reshape the result back to the original image dimensions
            D = D.reshape(nRows, nCols)

            # Display the result
            plt.figure(figsize=(10, 4))
            plt.imshow(D, cmap='Accent')  # Remove np.rot90 if not necessary
            plt.axis('off')
            plt.show()

            return D
        
        def export_data_to_csv(self):
            ordered_filepath = f'Output/ordered_minerals_{self.ROI_name}_technique{self.technique}_size_{self.num_samples}.csv'
            metrics_filepath = f'Output/technique_metrics_{self.ROI_name}_technique{self.technique}_size_{self.num_samples}.csv'
            
            ordered_data = []
            if self.technique == 'LASSO Regression':
                names = self.lasso_ordered_names
                coeffs = self.lasso_ordered_coefficients
            elif self.technique == 'Forward Stepwise Regression':
                names = self.forward_ordered_names
                coeffs = self.forward_ordered_coefficients
            elif self.technique == 'Mixed_Integer_Non-linear_Program':
                names = self.MINLP_ordered_names
                coeffs = self.MINLP_ordered_coefficients
            elif self.technique == 'HySudeB':
                names = self.HySudeB_ordered_names
                coeffs = self.HySudeB_ordered_coefficients

            # Combine the ordered name and coefficient data for the CSV
            for name, coeff in zip(names, coeffs):
                ordered_data.append([self.technique, name, coeff])

            # Write the ordered minerals and coefficients to CSV
            with open(ordered_filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Technique', 'Ordered Name', 'Coefficient'])  # Header
                writer.writerows(ordered_data)  # Data
            print(f"Ordered names and coefficients successfully exported to {ordered_filepath}")

            # Create data for technique metrics CSV
            metrics_data = []
            if self.technique == 'LASSO Regression':
                precision = self.avg_precision_lasso
                rmse = self.avg_RMSE_lasso
                runtime = self.avg_run_time_lasso
                model_size = self.avg_model_size_lasso
            elif self.technique == 'Forward Stepwise Regression':
                precision = self.avg_precision_forward
                rmse = self.avg_RMSE_forward
                runtime = self.avg_run_time_forward
                model_size = self.avg_model_size_forward
            elif self.technique == 'Mixed_Integer_Non-linear_Program':
                precision = self.avg_precision_MINLP
                rmse = self.avg_RMSE_MINLP
                runtime = self.avg_run_time_MINLP
                model_size = self.avg_model_size_MINLP
            elif self.technique == 'HySudeB':
                precision = self.avg_precision_HySudeB
                rmse = self.avg_RMSE_HySudeB
                runtime = self.avg_run_time_HySudeB
                model_size = self.avg_model_size_HySudeB

            # Add technique metrics data
            metrics_data.append([self.technique, precision, rmse, runtime, model_size])

            # Write the technique metrics to CSV
            with open(metrics_filepath, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Technique', 'Average Precision', 'Average RMSE', 'Run Time (s)', 'Model Size'])  # Header
                writer.writerows(metrics_data)  # Data
            print(f"Technique metrics successfully exported to {metrics_filepath}")

                    
                

                

                    
                