import numpy as np 
import pandas as pd
import spectral.io.envi as envi 
import matplotlib.pyplot as plt 
import random

if __name__ == '__main__':
    print("Running NNLS_Unmixing.py. Run from 'Run' notebook")
else:
    
    class chemphys:
        def __init__(self, spectral_library_filename):
            self.target_minerals = ['alunite', 'montmorillonite', 'kaolinite']
            self.load_new_spectral_library(spectral_library_filename)
            self.load_chemical_data()
            self.match_minerals_and_plot()

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

        def load_chemical_data(self):
            self.chemicaldf = pd.read_csv("mineraldata2.csv", encoding='latin1')  

        def match_minerals_and_plot(self):
            # Create a dictionary to hold spectral data categorized by mineral type
            mineral_types = self.chemicaldf['Category'].unique()
            mineral_dict = {mineral_type: [] for mineral_type in mineral_types}
            
            # Match the minerals in the spectral library with those in the chemical data
            for idx, name in enumerate(self.spectra_names):
                # Extract the mineral name after splitting by spaces and matching
                try:
                    mineral_key = name.split()[1]
                    matched_row = self.chemicaldf[self.chemicaldf['Name'] == mineral_key]
                    if not matched_row.empty:
                        mineral_type = matched_row.iloc[0]['Category']
                        mineral_dict[mineral_type].append((name, self.spectral_library[idx]))
                except IndexError:
                    print(f"WARNING: Unable to split and match name '{name}'")

            target_palette = ['purple', 'blue', 'green']
            for mineral_type, spectra in mineral_dict.items():
                if spectra:
                    plt.figure(figsize=(10, 6))
                    
                    # Separate target and non-target spectra
                    target_spectra = []
                    non_target_spectra = []
                    for name, spectrum in spectra:
                        if any(target in name.lower() for target in self.target_minerals):
                            target_spectra.append((name, spectrum))
                        else:
                            non_target_spectra.append((name, spectrum))
                    
                    # Randomly select up to 10 non-target spectra
                    selected_non_target_spectra = random.sample(non_target_spectra, min(10, len(non_target_spectra)))
                    
                    # Plot target spectra
                    for name, spectrum in target_spectra:
                        target_index = next(i for i, target in enumerate(self.target_minerals) if target in name.lower())
                        color = target_palette[target_index]
                        plt.plot(self.wavelengths, spectrum, label=name, color=color)
                    
                    # Plot selected non-target spectra
                    for name, spectrum in selected_non_target_spectra:
                        plt.plot(self.wavelengths, spectrum, label=name, color='lightgrey')
                    
                    plt.title(f'{mineral_type}')
                    plt.xlabel('Wavelength')
                    plt.ylabel('Reflectance')
                    plt.legend(bbox_to_anchor=(1, 1))
                    plt.show()



