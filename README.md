# DL-Course-Project-RED-CNN-Metal-Artifacts-Removal-in-CT



---

## Datasets
1. 1st dataset (MAYO)
should look like (as in ssinyu origin:)
```text
data_path
├── L067
│   ├── quarter_3mm
│   │       ├── L067_QD_3_1.CT.0004.0001 ~ .IMA
│   │       ├── L067_QD_3_1.CT.0004.0002 ~ .IMA
│   │       └── ...
│   └── full_3mm
│           ├── L067_FD_3_1.CT.0004.0001 ~ .IMA
│           ├── L067_FD_3_1.CT.0004.0002 ~ .IMA
│           └── ...
├── L096
│   ├── quarter_3mm
│   │       └── ...
│   └── full_3mm
│           └── ...      
...
│
└── L506
    ├── quarter_3mm
    │       └── ...
    └── full_3mm
            └── ...     
```
2. 2nd dataset (MAR)
should look like:
```text
data_path
├── train
│   ├── body5
│   │       ├── corrupted
│   │       │       ├── training_body_metalart_img4131_512x512x1.raw
│   │       │       ├── training_body_metalart_img4132_512x512x1.raw
│   │       │       └── ...
│   │       └── gt
│   │               ├── training_body_nometal_img4131_512x512x1.raw
│   │               ├── training_body_nometal_img4132_512x512x1.raw
│   │               └── ...
│   ├── body8
│   │       ├── corrupted
│   │       │       ├── training_body_metalart_img7131_512x512x1.raw
│   │       │       ├── training_body_metalart_img7132_512x512x1.raw
│   │       │       └── ...
│   │       └── gt
│   │               ├── training_body_nometal_img7131_512x512x1.raw
│   │               ├── training_body_nometal_img7132_512x512x1.raw
│   │               └── ...
│   └── body13
│           ├── corrupted
│           │       ├── training_body_metalart_img12050_512x512x1.raw
│           │       ├── training_body_metalart_img12051_512x512x1.raw
│           │       └── ...
│           └── gt
│                   ├── training_body_nometal_img12050_512x512x1.raw
│                   ├── training_body_nometal_img12051_512x512x1.raw
│                   └── ...
├── validation
│   ├── body5
│   │       └── ...
│   ├── body8
│   │       └── ...
│   └── body13
│         └── ...   
│
└── test
    ├── body5
    │       └── ...
    ├── body8
    │       └── ...
    └── body13
          └── ...  
```

---


## Prerequisites

This project was developed and tested in a Google Colab environment. If you are running this locally, ensure you have Python 3.8+ installed. 

Install the required dependencies using `pip`:

    pip install torch torchvision pydicom optuna numpy scipy matplotlib

**Core Dependencies:**
* **Deep Learning:** `torch`, `torchvision` (PyTorch) ,`torchview`
* **Medical Imaging & Vision:** `pydicom`, `scipy.ndimage`
* **Hyperparameter Optimization:** `optuna`
* **Data Manipulation & Plotting:** `numpy`, `matplotlib`
* **Standard Python Libraries:** `os`, `time`, `argparse`, `glob`, `collections`, `math`, `random`

*(Note: If you are running this inside Google Colab, PyTorch and basic data science libraries are pre-installed. You will only need to run `!pip install pydicom optuna` at the top of your notebook).*

---

## Repository Files

| File Name | Description |
| :--- | :--- |
| `Basic_UNet18.ipynb` |  |
| `Tailored_UNet18_MAYO_Dataset.ipynb` | Contains the implementation of the Residual Encoder-Decoder Convolutional Neural Network (RED-CNN) architecture. |
| `Tailored_UNet18_MAR_Dataset.ipynb` | Manages the training loop, loss calculation, optimization, and validation testing. |
| `` | Custom PyTorch `Dataset` and `DataLoader` classes for loading and batching the DICOM/Numpy data. |
| `` | Data preprocessing scripts (e.g., converting DICOM files to Numpy arrays, normalizing, and extracting patches). |
| `` | Contains evaluation metric functions to assess image quality, such as PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index). |
| `LICENSE` | MIT License detailing the permitted use and distribution of this project. |
| `README.md` | Project documentation, setup instructions, and acknowledgements. |

---

## Acknowledgements

* This project adapts components and architecture from the [RED-CNN PyTorch implementation](https://github.com/SSinyu/RED-CNN) by SSinyu. 
* Code modifications and hyperparameter tuning structures were developed with assistance from Google Gemini.

---

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.
