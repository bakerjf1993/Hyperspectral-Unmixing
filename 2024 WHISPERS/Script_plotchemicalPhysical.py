import numpy as np 
import pandas as pd
import spectral.io.envi as envi 
import matplotlib.pyplot as plt 
import random

if __name__ == '__main__':
    print("Running NNLS_Unmixing.py. Run from 'Run' notebook")
else:
    
    class chemphys:
        def __init__(self, spectral_library_filename, target_minerals=None):
            self.target_minerals = target_minerals if target_minerals else ['alunite', 'kaolinite']
            self.load_new_spectral_library(spectral_library_filename)
            self.load_chemical_data()
            self.match_minerals_and_plot()

        def load_new_spectral_library(self, spectral_library_filename):
            try:
                if spectral_library_filename.endswith('.hdr'):
                    lib = envi.open(spectral_library_filename)
                    self.wavelengths = lib.bands.centers
                    self.n_bands = len(self.wavelengths)
                    self.n_spectra = len(lib.names)
                    self.spectral_library = lib.spectra
                    self.spectra_names = lib.names  
                else:
                    raise ValueError("Library file should be a .hdr file!")
            except Exception as e:
                print(f"Error loading spectral library: {e}")

        def load_chemical_data(self):
            try:
                self.chemicaldf = pd.read_csv("mineraldata2.csv", encoding='latin1')
            except Exception as e:
                print(f"Error loading chemical data: {e}")

        def match_minerals_and_plot(self):
            # Create a dictionary to hold spectral data categorized by mineral type
            mineral_types = self.chemicaldf['Category'].unique()
            mineral_dict = {mineral_type: [] for mineral_type in mineral_types}
            
            # Match the minerals in the spectral library with those in the chemical data
            for idx, name in enumerate(self.spectra_names):
                try:
                    mineral_key = name.split()[1]
                    matched_row = self.chemicaldf[self.chemicaldf['Name'] == mineral_key]
                    if not matched_row.empty:
                        mineral_type = matched_row.iloc[0]['Category']
                        mineral_dict[mineral_type].append((name, self.spectral_library[idx]))
                except IndexError:
                    print(f"WARNING: Unable to split and match name '{name}'")

            self.plot_spectral_data(mineral_dict)

        def plot_spectral_data(self, mineral_dict):
            for mineral_type, spectra in mineral_dict.items():
                if spectra:
                    plt.figure(figsize=(10, 6))
                    
                    # Separate target and non-target spectra
                    alunite_spectrum = None
                    kaolinite_spectrum = None
                    non_target_spectra = []
                    for name, spectrum in spectra:
                        if 'alunite' in name.lower() and 'ammonio' not in name.lower() and alunite_spectrum is None:
                            alunite_spectrum = (name, spectrum)
                        elif 'kaolinite' in name.lower() and 'halloy' not in name.lower() and kaolinite_spectrum is None:
                            kaolinite_spectrum = (name, spectrum)
                        else:
                            non_target_spectra.append((name, spectrum))
                    
                    # Randomly select up to 10 non-target spectra
                    selected_non_target_spectra = random.sample(non_target_spectra, min(10, len(non_target_spectra)))
                    
                    if alunite_spectrum:
                        plt.plot(self.wavelengths, alunite_spectrum[1], label=alunite_spectrum[0], color='purple')

                        # Calculate RMSE between alunite spectrum and non-target spectra
                        rmse_list = []
                        for name, spectrum in non_target_spectra:
                            rmse = np.sqrt(np.mean((spectrum - alunite_spectrum[1])**2))
                            rmse_list.append((rmse, name, spectrum))
                            if 'mascagn2' in name.lower():
                                plt.plot(self.wavelengths, spectrum, label=name, linestyle='dotted', color='purple')

                        # Sort by RMSE and select top 10 unique spectra
                        rmse_list.sort()
                        selected_spectra = []
                        selected_names = set()

                        for rmse, name, spectrum in rmse_list[:5]:
                            if name not in selected_names:
                                selected_spectra.append((name, spectrum))
                                selected_names.add(name)

                        for rmse, name, spectrum in rmse_list[-5:]:
                            if name not in selected_names:
                                selected_spectra.append((name, spectrum))
                                selected_names.add(name)
                    
                    elif kaolinite_spectrum:
                        plt.plot(self.wavelengths, kaolinite_spectrum[1], label=kaolinite_spectrum[0], color='blue')
                    
                        rmse_list = []
                        selected_spectra = []
                        selected_names = set()

                        for name, spectrum in non_target_spectra:
                            rmse = np.sqrt(np.mean((spectrum - kaolinite_spectrum[1])**2))
                            rmse_list.append((rmse, name, spectrum))
                            if 'montmor' in name.lower():
                                plt.plot(self.wavelengths, spectrum, label=name, linestyle='dotted', color='blue')

                        # Sort by RMSE and select top 10 unique spectra
                        rmse_list.sort()                        

                        for rmse, name, spectrum in rmse_list[:5]:
                            if name not in selected_names:
                                selected_spectra.append((name, spectrum))
                                selected_names.add(name)

                        for rmse, name, spectrum in rmse_list[-5:]:
                            if name not in selected_names:
                                selected_spectra.append((name, spectrum))
                                selected_names.add(name)

                    # Plot selected non-target spectra
                    for name, spectrum in selected_non_target_spectra: 
                        if 'a-chlori' in name.lower():
                            plt.plot(self.wavelengths, spectrum, label=name, linestyle='dotted', color='purple')
                        elif 'hyperst2' in name.lower():
                            plt.plot(self.wavelengths, spectrum, label=name, linestyle='dotted', color='purple')
                        elif 'epidote1' in name.lower():
                            plt.plot(self.wavelengths, spectrum, label=name, linestyle='dotted', color='blue')
                        elif 'elbaite2' in name.lower():
                            plt.plot(self.wavelengths, spectrum, label=name, linestyle='dotted', color='blue')
                        else:
                            plt.plot(self.wavelengths, spectrum, label=name, color='darkgrey')
                    
                    plt.title(f'{mineral_type}')
                    plt.xlabel('Wavelength')
                    plt.ylabel('Reflectance')
                    plt.legend(ncol=2,bbox_to_anchor=(.95, -0.15))
                    plt.gcf().patch.set_alpha(0)
                    
                    plt.show()
