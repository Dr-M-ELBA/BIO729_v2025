# Spectroscopy Data

## Introduction
In our second example, we will apply EDA to spectroscopy data. Spectroscopy is a powerful analytical technique used to measure the interaction between matter and electromagnetic radiation.
It provides detailed information about the molecular composition, structure, and environment of a sample. It is so powerful that it is applied in almost every field, from analysing
composition of galaxies to studying microbes and their metabolites. 

In this example, we will work with spectroscopy data that was used to analyse cells. The labelling will unlikely make sense since it requires niche domain expertise. However, EDA protocol
should make sense. 

## Coding

First, install the relevant packages:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

import glob
import os
import zipfile
from google.colab import drive

from sklearn.decomposition import PCA
```
Same as before, leave a comment next to each package explaining its role. Also, is it clear how the packages are grouped?

Next, let's mount our Colab notebook to our Google Drive:

```python
drive.mount('/content/drive')
```

Then, let's unzip our file. Warning, there is a bug in the following code:

```python
unzip Cells_Raman_Spectra.zip
```

Although we unzipped the main folder, it contains zipped files that we will also need to unzip:

```pythong
zipped_files = glob.glob("Cells_Raman_Spectra/**/*.zip", recursive=True)
for file in zipped_files:
    with zipfile.ZipFile(file, 'r') as zip_ref:
        print(f"Extracting {file}...")
        zip_ref.extractall(os.path.dirname(file))
    os.remove(file)
```
The spectra are stored in .csv files, with each one containing 9 repeats. Let's inspect them:

```python
root_dir = "Cells_Raman_Spectra"
csv_files = glob.glob(os.path.join(root_dir, "*/*.csv"))

raman_data = {}
for file in csv_files:
    subfolder_name = os.path.basename(os.path.dirname(file))  
    csv_name = os.path.basename(file).replace(".csv", "")    
    label = f"{subfolder_name}-{csv_name}"                  

    raman_data[label] = pd.read_csv(file, header=None).values

for label, spectra in raman_data.items():
    plt.figure(figsize=(12, 8))  # Create a new figure for each file

    for i, spectrum in enumerate(spectra[1:], start=2): 
        plt.plot(spectrum, label=f"{label}-Row{i}") 

   
    plt.title(f"Raman Spectra from {label}", fontsize=16)
    plt.xlabel("Wavelength", fontsize=14)
    plt.ylabel("Intensity", fontsize=14)
    plt.legend(fontsize=10, loc="best")
    plt.grid(True)
    plt.show()
```

As seen, inspecting them individually is time-consuming and is unlikely to reveal any obvious differences. Does it make much sense? No, because each spectra contains over 100,000 datapoints. 
Superimposing all 400+ spectra does get complicated. In such cases, we can apply dimensionality reduction, whereby each spectrum is reduced from over 100,000 dimensions to just 2, 
whilst still retaining what makes them different. The code below applies principal component analysis (PCA; more on it later in the module):

```python
all_spectra = []  
group_labels = []  
csv_groups = list(raman_data.keys())  
for label, spectra in raman_data.items():
    all_spectra.extend(spectra[1:])  
    group_labels.extend([label] * (spectra.shape[0] - 1))  

all_spectra = []  
group_labels = []  
csv_groups = list(raman_data.keys()) 

for label, spectra in raman_data.items():
    all_spectra.extend(spectra[1:])  
    group_labels.extend([label] * (spectra.shape[0] - 1)) 

all_spectra = np.array(all_spectra)

scaler = MinMaxScaler()
normalized_spectra = scaler.fit_transform(all_spectra)

pca = PCA(n_components=2)
pca_result = pca.fit_transform(normalized_spectra)
pc1, pc2 = pca_result[:, 0], pca_result[:, 1]

unique_groups = sorted(set(group_labels))
num_groups = len(unique_groups)
colors = cm.tab20(np.linspace(0, 1, num_groups))  
group_colors = {group: color for group, color in zip(unique_groups, colors)}

plt.figure(figsize=(12, 8))
for group in unique_groups:
    indices = [i for i, label in enumerate(group_labels) if label == group]
    plt.scatter(pc1[indices], pc2[indices], label=group, color=group_colors[group], s=50)

plt.title("2D PCA of Normalized Raman Spectra", fontsize=16)
plt.xlabel("Principal Component 1", fontsize=14)
plt.ylabel("Principal Component 2", fontsize=14)
plt.legend(title="CSV Groups", fontsize=10, loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.grid(True)
plt.tight_layout()

plt.show()
```
As you will see, plotting the results of the PCA makes it easier to observe our spectra and identify any patterns. Can you improve the quality of the PCA plot?
