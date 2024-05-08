import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import spectral
import matplotlib.pyplot as plt
import seaborn as sns
import spectral.io.envi as envi
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    print("Running Unmixing.py. Run from 'Run' notebook")
else:
    
    class Unmix:
        def __init__(self, image_hdr_filename, image_filename, spectral_library_filename):
            # Load hyperspectral image and spectral library
            self.load_new_image(image_hdr_filename, image_filename)
            self.load_new_spectral_library(spectral_library_filename)

            # Initialize abundance_maps attribute
            self.abundance_maps = None
        
            # Print information about the loaded data
            print(f"n_bands: {self.n_bands}")
            print(f"n_spectra: {self.n_spectra}")
            print(f"spectral_library: {self.spectral_library}")
            print(f"n_rows: {self.n_rows}")
            print(f"n_cols: {self.n_cols}")
            print(f"n_imbands: {self.n_imbands}")
           
            # Plot wavelengths, set up colormap, and show the hyperspectral image
            self.plot_spectra()
            self.show_image()

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
            # Convert the 3D hyperspectral image arrayinto a 2D array
            reshaped_image = self.image_arr.reshape(self.n_rows * self.n_cols, self.n_bands)

            print(f"reshaped_image: {reshaped_image}")
            print(f"reshaped_image_shape: {reshaped_image.shape}")
            return reshaped_image

        def unmixing(self):
            # Change the shape of X to have spectra along rows and bands along columns
            X = self.spectral_library.T
            print(f"X: {X}")
            print(f"X_shape: {X.shape}")
            print("---------------------------------------------------------------")
            y = self.reshape_image()
            print("y: ",y)

            # Initialize arrays to store results
            pixel_models = []
            abundance_maps = np.zeros((self.n_rows*self.n_cols,self.n_spectra))

            # Iterate through each pixel spectrum
            for i,pixel_spectrum in enumerate(y):
                
                # Fit linear regression model
                model = LinearRegression(fit_intercept = False)
                model.fit(X,pixel_spectrum)

                # Get the coefficients
                abundance_coefficients = model.coef_

                # Ensure non-negativity: Modifies elements less than 0 to be exactly 0
                abundance_coefficients[abundance_coefficients < 0] = 0

                # Store the abundance map for each pixel spectrum
                abundance_maps[i, :] = abundance_coefficients

                # Append the model to the list
                pixel_models.append(model)

                if i%1000 == 0:
                    print(i)
                    
            # Reshape the abundance array
            abundance_array = abundance_maps.reshape(self.n_rows, self.n_cols,self.n_spectra)
            print(f"abundance_array_shape: {abundance_array.shape}")
           
            # Store the results in the class attributes for future use
            self.pixel_models = pixel_models
            self.abundance_maps = abundance_maps
            self.abundance_array = abundance_array 

            return abundance_array 
        
        def unmixing_small(self):
            # Extracts the first 10 spectra from the spectral library and transposes
            X = self.spectral_library[:10,:].T
            print(f"X: {X}")
            print(f"X_shape: {X.shape}")
            print("---------------------------------------------------------------")
            y = self.reshape_image()
            print("y: ",y)

            # Initialize arrays to store results
            abundance_maps = np.zeros((self.n_rows*self.n_cols,10))

            # Iterate through each pixel spectrum
            for i,pixel_spectrum in enumerate(y):
                
                # Fit linear regression model
                model = LinearRegression(fit_intercept = False, positive=True)
                model.fit(X,pixel_spectrum)

               # Get the coefficients
                abundance_coefficients = model.coef_

                # Ensure non-negativity: Modifies elements less than 0 to be exactly 0
                abundance_coefficients[abundance_coefficients < 0] = 0

                
                # Store the abundance map for each pixel spectrum
                abundance_maps[i, :] = abundance_coefficients

                if i%1000 == 0:
                    print(i)

            # Reshape the abundance array
            abundance_array = abundance_maps.reshape(self.n_rows, self.n_cols,10)
            print(abundance_array.shape)

            self.abundance_maps = abundance_maps
            self.abundance_array = abundance_array  
            return abundance_array 
        
        def select_index(self, index, library_indicies):
            

            # Extract the spectrum at the specified index
            if len(index) == 1:
                if index < 0 or index >= self.n_rows * self.n_cols:
                    raise ValueError("Invalid index. It should be between 0 and (n_rows * n_cols - 1).")
                pixel_spectrum = self.reshape_image()[index, :]
                print(f"pixel_spectrum.shape: {pixel_spectrum.shape}")
                
            else:
                pixel_spectrum = self.image_arr[index[0],index[1],:].flatten()
                print(f"pixel_spectrum.shape: {pixel_spectrum.shape}")
                   

            # Extract selected columns from the transposed spectral library
            X = self.spectral_library.T[:, library_indicies]
            print(f"X.shape: {X.shape}")
            

            # Convert X to a pandas DataFrame with column names
            X_df = pd.DataFrame(X, columns=[self.spectra_names[i] for i in library_indicies])

            
            # Fit linear regression model using OLS
            model = sm.OLS(pixel_spectrum, X_df)
            results = model.fit()

            # Ensure non-negativity of coefficients
            abundance_coefficients = results.params
            #abundance_coefficients[abundance_coefficients < 0] = 0

            

            # Print the normalized coefficients
            print("Abundance Coefficients:")
            print(abundance_coefficients)

            # Print the summary
            print("Model Summary:")
            print(results.summary())

            # Visualize the pixel spectrum and the estimated abundances
            plt.figure(figsize=(12, 7))
            plt.plot(self.wavelengths, pixel_spectrum, label="Pixel Spectrum",color="black")

            # Exclude "const" term from normalized_abundance before dot product
            plt.plot(self.wavelengths, np.dot(abundance_coefficients, X.T), label="Estimated Spectrum",color="red")

            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            plt.legend()
            plt.title(f'Unmixing Result for Pixel at Index {index}')
            plt.show()

            # Visualize the individual spectra used in unmixing
            plt.figure(figsize=(12, 7))
            for i in range(len(library_indicies)):
                plt.plot(self.wavelengths, X[:, i], label=self.spectra_names[library_indicies[i]])

            plt.plot(self.wavelengths, pixel_spectrum, label="Pixel Spectrum", color="black")
            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            plt.legend()
            plt.title(f'Unmixing Spectra')
            plt.show()

       
        def visualize_abundance_images(self, num_images=5):
            # Visualize abundance images for a few bands
            for i in range(min(num_images, self.n_spectra)):
                plt.figure(figsize=(12, 7))
                plt.imshow(self.abundance_array[:, :, i], cmap='jet')
                plt.title(f'Abundance for {self.spectra_names[i]}')
                plt.colorbar()
                plt.show()

        def visualize_abundance_map(self):
            if self.abundance_map is None:
                raise ValueError("Run unmixing() first to generate the abundance map.")

            # Reshape abundance_map if it's one-dimensional
            if len(self.abundance_map.shape) == 1:
                self.abundance_map = self.abundance_map.reshape(-1, 1)

            # Find the index of the maximum abundance for each pixel
            max_abundance_index = np.argmax(self.abundance_map, axis=1)

            # Create a color image based on the maximum abundance indices
            color_image = self.spectral_library[max_abundance_index]

            # Display the color image
            spectral.imshow(color_image)

        def plot_spectra(self, num_spectra=20):
            plt.figure(figsize=(12, 7))
            plt.grid(True)

            # Choose a subset of spectra and names to plot
            selected_spectra = self.spectral_library[:num_spectra]
            selected_names = self.spectra_names[:num_spectra]

            for spectra, name in zip(selected_spectra, selected_names):
                plt.plot(self.wavelengths, spectra, label=name)

            plt.xlabel('Wavelength')
            plt.ylabel('Reflectance')
            plt.legend()
            plt.title(f'Top {num_spectra} Spectra')
            plt.show()

        def show_image(self):
            plt.figure(figsize=(12, 7))
            plt.title("File_Image")
            skip = int(self.n_bands / 4)
            plt.imshow(self.image_arr[:, :, (skip, 2 * skip, 3 * skip)])

            

        
