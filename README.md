# MorphoGlia: Clustering and Mapping Microglia Morphology

- [I) Introduction](#i-introduction)
- [II) How to Start Using MorphoGlia](#ii-how-to-start-using-morphoglia)
  - [A) Interface Mode (user-friendly)](#a-interface-mode-user-friendly)
    - [1) Download the repository](#1-download-the-repository)
    - [2) Install dependencies via Conda](#2-install-dependencies-via-conda)
    - [3) Activate the environment](#3-activate-the-environment)
    - [4) Launch the interface](#4-launch-the-interface)
  - [B) Code Mode (For Developers)](#b-code-mode-for-developers)
  - [C) App Mode (Executable, under repair)](#c-app-mode-executable-under-repair)



![Graph_Abstract2](https://github.com/user-attachments/assets/c4b873ca-26cf-4715-b6ac-5041d964039a)


# I) Introduction


MorphoGlia has been developed with a focus on user-friendliness and accessibility, making it an ideal tool for the broader scientific community. The software is available in two main modes: a software mode and an interface mode, both designed to facilitate ease of use. The executable file was generated using PyInstaller (https://pyinstaller.org/en/stable/), while the interactive interface was built using the Tkinter Python library (https://docs.python.org/3/library/tkinter.html).
For advanced users, direct modification of the source code is recommended to tailor the application to the specific needs of individual experiments. This approach allows for greater flexibility and customization, ensuring that MorphoGlia can be adapted to a wide range of research scenarios.

To use the **interface mode**, download the MorphoGlia_Interface directory. You can run it from the terminal or through a Python interface. Ensure that the files MorphoGlia_app.py and morphoglia.py are located in the same directory. You will need to install the necessary libraries.

The **executable mode** is available for download at the following link: https://drive.google.com/drive/u/1/folders/15Mu2THZvVH6OTlDZuzf7ftsWyLAiWjUV

The **source code** is located in the MorphoGlia_code directory. This directory includes the interactive mode for point tracking, color mapping to customize cluster colors, and the R and Python scripts needed to reproduce the graphics.

Also, check out the video **"MorphoGlia Tutorial" on YouTube** for an example of how to use the MorphoGlia software: https://www.youtube.com/watch?v=OLLS9I8ln48&t=17s

Currently, the software is available **for Macs with M1/M2 processors**. We are working on versions for Intel-based Macs and Windows.

# II) How to start using MorphoGlia



## A) Interface Mode (user-friendly)

This is the most user-friendly way to run MorphoGlia using its graphical interface. It is ideal for new users and does not require writing any code.

### 1) Download the repository

1. Visit the repository:  
   https://github.com/Maya-Arteaga/MorphoGlia  
2. Click the green **"Code"** button at the top right.  
3. Select **"Download ZIP"**.  
4. Unzip the downloaded file to a location of your choice.



![Download_4](https://github.com/user-attachments/assets/c497a4b7-8846-4a6f-996d-b3cd9ab5e38d)
---

### 2) Install dependencies via Conda

After unzipping the repository:

1. Locate the file named `morphoglia.yml` inside the folder.
2. Open a terminal (see below).
3. Make sure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed.
4. Navigate to the folder containing the `.yml` file. Example:

 ```bash
   cd ~/Downloads/MorphoGlia-main/MorphoGlia_Interface
```

5. Create the environment by running:

```bash
   conda env create -f morphoglia.yml
```


If you encounter issues with the one-step installation, you can install each library manually

**Manual Installation:**
If you encounter issues with the one-line installation, you can install the libraries manually. Just copy and paste the following commands into your terminal one by one, skipping any lines that start with # (comments). Make sure Conda is already installed.

```bash
# Manual Installation:

# Step 1: Create a new Conda environment with Python 3.10.14
conda create -n morphoglia python=3.10.14 -y

# Step 2: Activate the new enviroment
conda activate morphoglia

# Step 3: Install OpenCV (via conda-forge for version control)
conda install -c conda-forge opencv=4.10.0 -y

# Step 4: Install the rest via pip in one line
pip install \
    pandas==2.2.2 \
    tifffile==2024.8.10 \
    scikit-image==0.24.0 \
    matplotlib==3.9.2 \
    scikit-learn==1.5.1 \
    seaborn==0.13.2 \
    umap-learn==0.5.6 \
    hdbscan==0.8.38.post1 \
    datashader==0.16.3 \
    bokeh==3.5.1 \
    holoviews==1.19.1


# Step 5: Verify correct installation
python -c "import cv2, pandas as pd, tifffile, skimage, matplotlib, sklearn, seaborn as sns, umap, hdbscan, datashader, bokeh, holoviews; print('✔', cv2.__version__, pd.__version__, tifffile.__version__, skimage.__version__, matplotlib.__version__, sklearn.__version__, sns.__version__, umap.__version__, hdbscan.__version__, datashader.__version__, bokeh.__version__, holoviews.__version__)"

```




6. Verify the environment was created:

```bash
   conda env list
```
You should see morphoglia listed in the output.

![2_install](https://github.com/user-attachments/assets/bdbb45d5-76f1-44de-915b-26c82dab976a)

### 3) Activate the environment

Activate the newly created environment by running:

```bash
   conda activate morphoglia
```
If successful, your terminal prompt should now show (morphoglia).

### 4) Launch the Interface

Once inside the morphoglia environment, navigate to the interface folder (if not already there):


```bash
   cd ~/Downloads/MorphoGlia-main/MorphoGlia_Interface
```
Then run:

```bash
   python main.py
```

This will launch the MorphoGlia graphical interface.



![Gif_python](https://github.com/user-attachments/assets/5afc5741-8a9f-4189-8bb3-7d98716225ac)













