# MorphoGlia: Clustering and Mapping Microglia Morphology

- [I) Introduction](#I-introduction)
- [II) How to Install MorphoGlia](#II-how-to-start-using-MorphoGlia)


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

## 1) Python Environment Setup

Ensure you have **Python 3.10.14** installed. You can use a terminal or an Integrated Development Environment (IDE) such as Spyder, PyCharm, or Visual Studio.

## 2) Install Required Libraries

To ensure compatibility, install the following libraries and verify their versions:

```bash
# Install OpenCV
pip install opencv-python
python -c "import cv2; print(cv2.__version__)"  # Expected: 4.10.0

# Install Pandas
pip install pandas
python -c "import pandas as pd; print(pd.__version__)"  # Expected: 2.2.2

# Install TiffFile
pip install tifffile
python -c "import tifffile; print(tifffile.__version__)"  # Expected: 2024.8.10

# Install Scikit-Image
pip install scikit-image
python -c "import skimage; print(skimage.__version__)"  # Expected: 0.24.0

# Install Matplotlib
pip install matplotlib
python -c "import matplotlib; print(matplotlib.__version__)"  # Expected: 3.9.2

# Install Scikit-Learn
pip install scikit-learn
python -c "import sklearn; print(sklearn.__version__)"  # Expected: 1.5.1

# Install Seaborn
pip install seaborn
python -c "import seaborn as sns; print(sns.__version__)"  # Expected: 0.13.2

# Install UMAP-learn
pip install umap-learn
python -c "import umap; print(umap.__version__)"  # Expected: 0.5.6

# Install HDBSCAN
pip install hdbscan
pip show hdbscan | grep Version  # Expected: 0.8.38.post1

# Install Datashader
pip install datashader
pip show datashader | grep Version  # Expected: 0.16.3

# Install Bokeh
pip install bokeh
pip show bokeh | grep Version  # Expected: 3.5.1

# Install HoloViews
pip install holoviews
pip show holoviews | grep Version  # Expected: 1.19.1

```

## 3) Downloading and Running Files


![Download_4](https://github.com/user-attachments/assets/c497a4b7-8846-4a6f-996d-b3cd9ab5e38d)


**a)** Open the `MorphoGlia-main` folder and move the `MorphoGlia_Interface` folder to your Desktop.

**b)** Open your terminal and navigate to the directory containing the `Morphoglia_Interface` files. For example:

Locate on the directory "Morphoglia_Interface". For example:
```bash
cd Desktop/Morphoglia_Interface
```

**c)** Run the interface using the following command:

```bash
python Morphoglia_app.py
```

This will start and display the interface mode. 

Alternatively, you can run the interface from a Python IDE by opening the Morphoglia_app.py file and executing it.

![Gif_python](https://github.com/user-attachments/assets/5afc5741-8a9f-4189-8bb3-7d98716225ac)













