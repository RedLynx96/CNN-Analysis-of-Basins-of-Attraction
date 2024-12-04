
# 🌀 Deep Learning-Based Analysis of Basins of Attraction 🌀

This repository contains the tools and resources for analyzing basins of attraction using deep learning techniques. We explore fractal dimension, basin entropy, boundary basin entropy, and the Wada property through neural network-based methods. Feel free to explore our work, where we provide an in depth explanation of our findings.

---

## 🔧 Setting Up the Environment

To run the code provided in this repository, we recommend using Anaconda to create a dedicated Python environment.

1. Clone this repository:
   ```bash
   git clone https://github.com/RedLynx96/CNN-Analysis-of-Basins-of-Attraction.git

   cd CNN-Analysis-of-Basins-of-Attraction
   ```

2. Create the Conda environment using the provided `environment.yml` file:
   ```bash
   cd envs
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate Basins_CNN
   ```
---

## 📚 Available Scripts

### 1️⃣ **Basin_predictor.py**
This script provides a GUI to compute metrics for basins of attraction from a `.csv` file containing precomputed basins.

- **Usage**:  
  - Open the GUI by running the script:
    ```bash
    python Basin_predictor.py
    ```
  - Load a `.csv` file with the basins' routes, an example folder and dataset is provided.  
  - The script calculates metrics such as fractal dimension, basin entropy, and more.  
  - Results are saved in a new `.csv` file.

- **Visualization**:  
  - Basins can be visualized within the GUI. A demonstration GIF is included in the repository for reference.

  <div align="center">
    <img src="media/Example.gif" alt="Demo of Basin_predictor.py">
  </div>
---

### 2️⃣ **Basin_Metric_Training.py**
This script allows training a new CNN model.  

**Note**: Training a new model requires a large dataset of basins. Pre-trained weights are provided in `models/`.

- **Usage**:  
  - Train a new model:
    ```bash
    python Basin_Metric_Training.py
    ```
  - Use the pre-trained weights for analysis:
    - Weights are available in `models/`.

---

## ✨ Features

- **Automated Metrics Calculation**: Compute basin metrics like fractal dimension and Wada property.  
- **Visualization Tools**: Explore basins of attraction visually.  
- **Pre-trained Models**: Use our ResNet50 model for rapid analysis.  
- **Custom Training**: Train new models with your datasets.  

---

## 🌐 Resources

- **Datasets and Models**:  
  Find the datasets and pre-trained models at [Zenodo](https://zenodo.org/records/10550982).

- **Research Papers**:  
  - [Characterization of Fractal Basins Using Deep Convolutional Neural Networks](https://doi.org/10.1142/S0218127422502005).  
  Published in *International Journal of Bifurcation and Chaos* (2022).  
  - [Deep learning-based analysis of basins of attraction](https://doi.org/10.1063/5.0159656).  
  Published in *Chaos* (2024).

---

## 📜 Reference

If you use this code in your work, please cite our paper:

**David Valle, Alexandre Wagemakers, Miguel A. F. Sanjuán; *Deep learning-based analysis of basins of attraction*.**  
Published in [Chaos, March 2024 (Vol. 34, Issue 3)](https://doi.org/10.1063/5.0159656).  
DOI: [10.1063/5.0159656](https://doi.org/10.1063/5.0159656)

**David Valle, Alexandre Wagemakers, Alvar Daza, Miguel A. F. Sanjuán; Characterization of Fractal Basins Using Deep Convolutional Neural Networks.**   
Published in [International Journal of Bifurcation and Chaos, 2022 (Vol. 32, Issue 13)](https://doi.org/10.1142/S0218127422502005).   
DOI: [10.1142/S0218127422502005](https://doi.org/10.1142/S0218127422502005)

```bibtex
@article{valle2022characterization,

  title = {Characterization of Fractal Basins Using Deep Convolutional Neural Networks},
  author = {David Valle, Alexandre Wagemakers, Alvar Daza, Miguel A. F. Sanjuán},
  journal = {International Journal of Bifurcation and Chaos},
  volume = {32},
  number = {13},
  year = {2022},
  doi = {10.1142/S0218127422502005},
  url = {https://doi.org/10.1142/S0218127422502005}
}

@article{valle2024deep,
  title={Deep learning-based analysis of basins of attraction},
  author={David Valle, Alexandre Wagemakers, Miguel A. F. Sanjuán},
  journal={Chaos},
  volume={34},
  number={3},
  pages={033105},
  year={2024},
  publisher={AIP Publishing},
  doi={10.1063/5.0159656}
}
```
