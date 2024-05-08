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

if __name__ == '__main__':
    print("Running Summary_Compare.py. Run from 'Run' notebook")
else:
    
    class COMPARE:
        def __init__(self, image_hdr_filename, image_filename, spectral_library_filename):
            self.load_new_image(image_hdr_filename, image_filename)
            self.load_new_spectral_library(spectral_library_filename)
            #f, axarr = plt.subplots(1,2,figsize=(15, 15))
            #axarr[0].imshow(self.show_image())
            #axarr[1].imshow(self.show_paper_im())

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

        def summary(self,image_hdr_filename,image_filename,spectral_library_filename,pixel_index, alpha_lasso=1, alpha_ridge=0.5):
            ols_unmixer = OLS(image_hdr_filename, image_filename, spectral_library_filename)
            ols_result = ols_unmixer.selectedindex_fit(y_index=pixel_index)
            ols_plt = ols_unmixer.plot_spectra()
            OLS_summary = ols_unmixer.final_summary()  

            print("---------------------------------")
            nnls_unmixer = NNLS(image_hdr_filename, image_filename, spectral_library_filename)
            nnls_result = nnls_unmixer.selectedindex_fit(y_index=pixel_index)
            NNLS_plt = nnls_unmixer.plot_spectra()
            NNLS_summary = nnls_unmixer.final_summary()

            print("---------------------------------")
            lasso_unmixer = LASSO(image_hdr_filename, image_filename, spectral_library_filename)
            lasso_result = lasso_unmixer.selectedindex_fit(y_index=pixel_index, alpha=alpha_lasso)
            lasso_plt = lasso_unmixer.plot_spectra()
            lasso_summary = lasso_unmixer.final_summary()

            print("---------------------------------")
            ridge_unmixer = RIDGE(image_hdr_filename, image_filename, spectral_library_filename)
            ridge_result = ridge_unmixer.selectedindex_fit(y_index=pixel_index, alpha=alpha_ridge)
            ridge_plt = ridge_unmixer.plot_spectra()
            ridge_summary = ridge_unmixer.final_summary()

            print("---------------------------------")
            step_wise_unmixer = STEPWISE(image_hdr_filename, image_filename, spectral_library_filename)
            step_forward_result = step_wise_unmixer.forward_selectedindex_fit(y_index=pixel_index)
            step_forward_plt = step_wise_unmixer.plot_spectra()
            step_forward_summary = step_wise_unmixer.final_summary()
            print("---------------------------------")
            step_back_result = step_wise_unmixer.backward_selectedindex_fit(y_index=pixel_index)
            step_back_plt = step_wise_unmixer.plot_spectra()
            step_back_summary = step_wise_unmixer.final_summary()

            print("---------------------------------")
            BMA_unmixer = BMA(image_hdr_filename, image_filename, spectral_library_filename)
            BMA_result = BMA_unmixer.selectedindex_fit(y_index=pixel_index, MaxVars=5)
            BMA_plt = BMA_unmixer.plot_spectra()
            BMA_summary = BMA_unmixer.final_summary()
            BMA_summary_c = BMA_unmixer.final_summary_compare()
            
            print("---------------------------------")
            summary_df  = pd.DataFrame(columns=['Method', 'Spectra', 'Abundance'])
            for name,summary in [["OLS", OLS_summary],["NNLS", NNLS_summary],["LASSO", lasso_summary],["RIDGE",ridge_summary],\
                                 ["STEP_F",step_forward_summary],["STEP_B",step_back_summary],["BMA",BMA_summary_c]]:
                for s,a in summary[0:10]:
                    summary_df.loc[len(summary_df.index)] = [name, s, a]             

            
            display(summary_df)
            return summary_df

        def show_paper_im(self):
            im = img.imread('cuperite_paper.png') 
            plt.imshow(im) 
            return im

        def show_image(self,pix=[]):
            
            plt.title("File_Image")
            skip = int(self.n_bands / 4)
            imRGB = np.zeros((self.n_rows,self.n_cols,3))
            for i in range(3):
                imRGB[:,:,i] = self.stretch(self.image_arr[:,:,i * skip])
            for loc in pix:
                imRGB[loc[0],loc[1],0] = 1
                imRGB[loc[0],loc[1],1] = 0
                imRGB[loc[0],loc[1],2] = 0
            plt.imshow(imRGB)
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

