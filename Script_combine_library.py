import numpy as np
import pandas as pd
import spectral.io.envi as envi
import matplotlib.pyplot as plt


class Import_lib:
    def __init__(self, spectral_library_filename):
        self.load_new_spectral_library(spectral_library_filename)
        
        # Print information about the loaded data
        print(f"n_bands: {self.n_bands}")
        print(f"n_spectra: {self.n_spectra}")
        print(f"spectral_library: {self.spectral_library}")
        
        self.plot_spectra()

    def load_new_spectral_library(self, spectral_library_filename):
        if spectral_library_filename.endswith('.hdr'):
            # Open the ENVI header file
            lib = envi.open(spectral_library_filename)

            # Wavelength values extracted from the bands of the library
            self.wavelengths = lib.bands.centers

            # The number of bands (wavelengths) in the spectral library
            self.n_bands = len(self.wavelengths)

            # The number of spectra (materials) in the spectral library
            self.n_spectra = len(lib.names)

            # The actual spectral data (reflectance or radiance values)
            self.spectral_library = lib.spectra

            # The names of the spectra
            self.spectra_names = lib.names  # Update spectra_names with material names
        else:
            print("WARNING: library should be a .hdr file!")

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

class combine:
    def __init__(self, spectral_library_filenames):
        self.spectral_libraries = []
        self.comninelibraries(spectral_library_filenames)
        
        self.plot_combined_spectra()
        self.print_library_info()

    def comninelibraries (self, spectral_library_filenames):
        for filename in spectral_library_filenames:
            if filename.endswith('.hdr'):
                # Open the ENVI header file
                lib = envi.open(filename)

                # Add the loaded library to the list
                self.spectral_libraries.append({
                    'wavelengths': lib.bands.centers,
                    'spectra': lib.spectra,
                    'names': lib.names
                })
            else:
                print(f"WARNING: {filename} should be a .hdr file!")

    def print_library_info(self):
        for idx, lib in enumerate(self.spectral_libraries):
            n_bands = len(lib['wavelengths'])
            n_spectra = len(lib['names'])
            print(f"Library {idx + 1}:")
            print(f"  n_bands: {n_bands}")
            print(f"  n_spectra: {n_spectra}")

    def plot_combined_spectra(self, num_spectra=20):
        plt.figure(figsize=(12, 7))
        plt.grid(True)

        for idx, lib in enumerate(self.spectral_libraries):
            wavelengths = lib['wavelengths']
            spectra = lib['spectra']
            names = lib['names']

                      
            selected_spectra = spectra[:num_spectra]
            selected_names = names[:num_spectra]
            

            for spectra, name in zip(selected_spectra, selected_names):
                plt.plot(wavelengths, spectra, label=f"Library {idx + 1}: {name}")

        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        #plt.legend()
        plt.title(f'Top {num_spectra} Spectra from Combined Libraries')
        plt.show()