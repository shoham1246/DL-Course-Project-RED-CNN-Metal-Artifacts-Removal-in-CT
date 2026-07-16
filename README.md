# Analyzing CT Images Enhancement with Deep Networks

## 1. Project Background
The goal of this project is to analyze and compare deep neural network implementations for the enhancement of CT images containing various types of noise, utilizing Google Colab for model training, evaluation, and analysis. 

Computed Tomography (CT) images are widely used in the medical field, and their accurate analysis is crucial for saving human lives. However, these images can suffer from defects that degrade their quality. In this project, we focused on two primary architectures based on the Encoder-Decoder structure:
* **RED-CNN**: A Residual Encoder-Decoder Convolutional Neural Network evaluated initially on noisy CT images.
* **UNet18**: An architecture based on UNet, utilizing ResNet18 as the encoder.

The **RED-CNN** structure illustration:

<img width="586" height="283" alt="image" src="https://github.com/user-attachments/assets/daf734b4-3f6d-4a37-9cba-f3e0b53d7376" />

The **UNet18** structure illustration:

<img width="360" height="707" alt="image" src="https://github.com/user-attachments/assets/e9a6e758-96bb-48fb-ac93-d3cd308f0b92" />


We evaluated the performance of these models on two distinct datasets representing different defects: low-dose radiation noise (Poisson noise) and artifacts originating from artificial metal implants. Our analysis covers their generalization capabilities, advantages, disadvantages, and the impact of training strategies and hyperparameter tuning.

---

## 2. Data Sets
This project uses two open-source medical CT datasets:
1. **MAYO Data Set**: Contains Quarter-Dosed (QD) noisy CT images as inputs and Full-Dosed (FD) clean CT images as Ground Truth (GT).
2. **MAR Data Set**: Contains CT images with artificial metal implant artifacts as inputs, and clean images without artifacts as the Ground Truth.

### Internal Division
To ensure a fair comparison, both datasets had almost the same samples amount, and they were divided similarly. The division is as follows:

| Data Set | Division Logic | Train Set | Validation Set | Test Set |
| :--- | :--- | :--- | :--- | :--- |
| **MAYO** | Each of the 10 patients was assigned to a specific subset. | 87% | 5% | 8% |
| **MAR** | Each of the 3 body parts was divided according to the same rates. | 87% | 5% | 8% |

### Data Sets Directories Hierarchy

Detailed explanations could be found in the `READ ME files for data tracking` folder.

1. MAYO data set

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
2. MAR data set

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
    │       └── ...
    └──test_masks
            └── ...  
```
---

## 3. Models
All final models weights we received for M1, M2, .., M11 are found in a drive folder, the link is: https://drive.google.com/drive/folders/1LVmtOIGb199hCAvfnNAkfAfqPPcauit5?usp=sharing .

## 4. API
The following arguments can be passed to configure the training and testing phases in the `main` execution of the notebooks:

| Argument | Description |
| :--- | :--- |
| `--mode` | Operational running mode: `train`, `test`, `train_decoder`, or `train_fine_tune`. |
| `--load_mode` | `0` (slow load from disk) or `1` (fast load into RAM if memory permits). |
| `--data_path` | Directory path where the raw dicom/raw dataset is stored. |
| `--saved_path` | Directory path to save/load pre-processed `.npy` array files. |
| `--save_path` | Directory path to save model weights (`.ckpt`), logs, and figures. |
| `--result_fig` | Boolean; whether to save and display matplotlib comparison figures during testing. |
| `--norm_range_min` | Minimum Hounsfield Unit (HU) limit for normalization. |
| `--norm_range_max` | Maximum Hounsfield Unit (HU) limit for normalization. |
| `--trunc_min` | Minimum HU limit for truncation during testing. |
| `--trunc_max` | Maximum HU limit for truncation during testing. |
| `--patch_n` | Number of patches to extract from a single full image during training. |
| `--patch_size` | The spatial size (H x W) of the extracted square patches, integer scalar. |
| `--batch_size` | Number of samples per batch. |
| `--num_epochs` | Total number of training epochs. |
| `--print_iters`| Integer; determines in how many iterations will print the progress, train loss and time in the trainning . |
| `--decay_iters`| Integer; determines in how many iterations should the base learning rate drop in half. |
| `--save_iters`| Integer; determines in how many epoch the model's weights should be saved. |
| `--test_iters` | Integer; used to specify which checkpoint to load, indicates the number of iterations in the .ckpt file (e.g.: REDCNN_12000iter.ckpt, test_iters = 12000). Use this argument to fine-tune the model in training ('train_decoder' or 'train_fine_tune' modes) or to test a model with these weights.
| `--lr` | Base learning rate for the Adam optimizer. |
| `--lr_encoder` | Specific learning rate for encoder layers (overrides `--lr` if set). |
| `--lr_decoder` | Specific learning rate for decoder layers (overrides `--lr` if set). |
| `--decay_type` | Type of learning rate scheduler (`none`, `step`, `cosine`). (defaults to none) |
| `--warmup_epochs` | Number of epochs for linear learning rate warmup. |
| `--device` | Set explicitly to `cuda` or `cpu` (defaults to auto-detect). |
| `--multi_gpu` | Boolean; determines if the user has multiple GPU's, if there are multiple GPU's than the code will use `DataParallel` to split the workload. (defaults to False)
| `--num_workers` | Integer; determines how many separate processes will handle data loading in parallel. (defaults to 2) |
| `--ismask` | Boolean; determines whether to compute metrics only in the artifact ROI masks. |

---

## 5. Prerequisites
The project was developed and run on Google Colab. The primary dependencies are:

| Python Library | Version (Google Colab Default/Used) |
| :--- | :--- |
| `python` | 3.10+ |
| `torch` | 2.2+ |
| `torchvision` | 0.17+ |
| `numpy` | 1.25+ |
| `matplotlib` | 3.7+ |
| `pydicom` | 3.0.2 |
| `scipy` | 1.11+ |
| `optuna` | 3.6+ |

*Note: You can install missing dependencies via `!pip install pydicom optuna`.*

---

## 6. Files in the Repository

| Parent Folder | File Name | Description |
| :--- | :--- | :--- |
| RED-CNN | `RED_CNN_MAYO_M1.ipynb` | Implementation of the RED-CNN architecture adapted for the MAYO low-dose CT dataset. |
| RED-CNN | `RED_CNN_MAR_M2M3M4.ipynb` | Implementation of the RED-CNN architecture adapted for the MAR metal artifact dataset. |
| RED-CNN | `RED_CNN_Implementation_Results.ipynb` | Detailed Results of experiments concerning RED-CNN.|
| UNet18 | `Basic_UNet18_MAYO_M5.ipynb` | Implementation of the standard U-Net architecture using an untrained ResNet18 encoder adapted to the MAYO low-dose CT dataset. |
| UNet18 | `Tailored_UNet18_MAYO_M6.ipynb` | Tailored UNet18 implementation adapted to the MAYO low-dose CT dataset, utilizes ImageNet pre-trained weights, customized learning rates, 5-epoch warm up and 256x256 patch sizes. |
| UNet18 | `Tailored_UNet18_MAR_M7_to_M11.ipynb` | Implementation of the Tailored UNet18 architecture adapted for the MAR metal artifact dataset. |
| UNet18 | `UNet18_Model_Results.ipynb` | Detailed Results of experiments concerning UNet18.| 
| General | `Demonstrate_CT_Datasets.ipynb` | A demonstration of pair of ground truth and noisy input image from the MAYO low-dose CT dataset and the MAR metal artifact dataset. |
| General | `Graphics_Results_Producing.ipynb` | Helper functions to generate side-by-side metric comparison graphs (Loss, LR, PSNR, SSIM, RMSE) for single or multiple models. |
| READ ME files for data tracking | READ ME MAYO.txt | Explanations of the use of the MAYO low-dose CT dataset in this project. |
| READ ME files for data tracking | READ ME MAR.txt | Explanations of the use of the MAR metal artifact dataset in this project. | 
| - | `LICENSE` | MIT permissive license with conditions only requiring preservation of copyright and license notices. |
| - | `README.md` | The current file, explanation and documentation of this project. |
---

## 7. Experiments, Analysis and Graphic Results Guidelines
To replicate the experiments discussed in the report using Google Colab, follow these guidelines. You can access all the models (M1, M2, ..) from the link we attached above.

### Experiment 1: Creating Workframe, Restoring Results from Literature
* **M1 (Mayo RED-CNN):** Open `RED_CNN_MAYO_M1.ipynb`. In cell 7, set `--mode` to `train`, then run cells 0-7. To test your model, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that the saved model .ckpt file is the only file with this name in the `save` folder.  The results will be shown in the output of cell 7.
* **Analysis 1 (M1 Generalization):** Open `RED_CNN_MAR_M2M3M4.ipynb`. In cell 7, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that M1 saved model (REDCNN_12000iter.ckpt) file is the only file with this name in the `save` folder.  The results will be shown in the output of cell 7.

### Experiment 2: Examining UNet18 Variants
* **M5 (Mayo UNet18):** Open ` Basic_UNet18_MAYO_M5.ipynb`. In cell 7, set`--mode` to `train`, then run cells 0-7. To test your model, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that the saved model .ckpt file is the only file with this name in the `save` folder. The results will be shown in the output of cell 7.
* **M6 (Mayo Tailored UNet18):** Open `Tailored_UNet18_MAYO_M6.ipynb`. In cell 7, set `--mode` to train, then run cells 0-7. To test your model, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that your saved model .ckpt file is the only file with this name in the `save` folder. The results will be shown in the output of cell 7.

### Experiment 3: Adjusting Workframe and Examining Different Initializations – Part A
Open `RED_CNN_MAR_M2M3M4.ipynb`.
* **M2 (MAR RED-CNN):** In cell 7, set `--mode` to `train`, then run cells 0-7. To test your model, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that the saved model .ckpt file is the only file with this name in the `save` folder.  The results will be shown in the output of cell 7.
* **M3 (M1 Fine-Tuned RED-CNN on MAR):** Run cells 0-7. To train the model, upload M1 weights .ckpt file to the `save` folder created in cell 7, in cell 7 set `--test_iters` to the number of iterations from the saved M1 weights (in this case 12000). Please pay attention that this is the only .ckpt file with that name in the `save` folder.  Set `--mode` to `train_fine_tune`and run cell 7. To test your model, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that your saved model .ckpt file is the only file with this name in the `save` folder. The results will be shown in the output of cell 7.
* **M4 (M1 Fine-Tuned decoder RED-CNN on MAR):** Run cells 0-7. To train the model, upload M1 weights .ckpt file to the `save` folder created in cell 7, in cell 7 set `--test_iters` to the number of iterations from the saved M1 weights (in this case 12000). Please pay attention that this is the only .ckpt file with that name in the `save` folder.  Set `--mode` to `train_decoder`and run cell 7. To test your model, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that your saved model .ckpt file is the only file with this name in the `save` folder. The results will be shown in the output of cell 7.
* **Analysis 2 (M6 Generalization):** Open`Tailored_UNet18_MAR_M7_to_M11.ipynb`. In cell 7, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that M6 saved model (ResNet18UNet_4000iter.ckpt) file is the only file with this name in the `save` folder.  The results will be shown in the output of cell 7.

### Experiment 4: Examining Different Initializations – Part B
Open `Tailored_UNet18_MAR_M7_to_M11.ipynb`.
* **M7 (ImageNet Fine-tuned UNet18 on MAR):** To train the model, in cell 7, set `--mode` to `train` and run cells 0-7. To test your model, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that your saved model .ckpt file is the only file with this name in the `save` folder. The results will be shown in the output of cell 7.

* **M8 (M6 Fine-tuned Unet18 on MAR):** Run cells 0-7. To train the model, upload M6 weights .ckpt file to the `save` folder created in cell 7, in cell 7 set `--test_iters` to the number of iterations from the saved M6 weights (in this case 4000). Please pay attention that M6 saved model (ResNet18UNet_4000iter.ckpt) file is the only file with that name in the `save` folder.  Set `--mode` to `train_fine_tune` and run cell 7. To test your model, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that your saved model .ckpt file is the only file with this name in the `save` folder. The results will be shown in the output of cell 7.

### Experiment 5: Utilizing Optuna & Training Strategies
Open `Tailored_UNet18_MAR_M7_to_M11.ipynb`.
* **M9 (M6 Fine-tuned Unet18 on MAR with Optuna):** To use Optuna to find the optimal LRs for encoder and decoder, before running the code, in cell 7 change `--mode` to `optuna `than run cells 0-7. Later to train the model with the LRs, pass the LRs that you got in the output of cell 7 to `--lr_encoder` (we got: 2.7589e-05) and `--lr_decoder`(we got: 0.0012). In the `save` folder created in cell 7, upload M6 weights .ckpt file, set `--test_iters` to the number of iterations from the saved M6 weights (in this case 4000). Please pay attention that M6 saved model (ResNet18UNet_4000iter.ckpt) file is the only file with that name in the `save` folder.  Set `--mode` to `train_fine_tune`and run cell 7. To test your model, set `--mode` to  `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that your saved model .ckpt file is the only file with this name in the `save` folder. The results will be shown in the output of cell 7.
* **M10 (M6 Fine-tuned Unet18 on MAR with Optuna and Cosine sched.):** Run cells 0-7. In the `save` folder created in cell 7, upload M6 weights .ckpt file, set in cell 7 `--test_iters` to the number of iterations from the saved M6 weights (in this case 4000). Please pay attention that M6 saved model (ResNet18UNet_4000iter.ckpt) file is the only file with that name in the `save` folder.  Set `--mode` to `train_fine_tune`, switch `--decay_type` to `cosine`, pass the LRs `--lr_encoder`(we got: 2.7589e-05) and `--lr_decoder`(we got: 0.0012) and run cell 7. To test your model, set `--mode` to `test` and run cell 7 (and cells 0-6, if they weren’t run yet), please pay attention that your saved model .ckpt file is the only file with this name in the `save` folder. The results will be shown in the output of cell 7.
* **M11 (M6 Fine-tuned Unet18 on MAR with Optuna and Augment.):** In cell 5, in the definition of `get_patch`, set `augment=True`, then run cells 0-7. In the `save` folder created in cell 7, upload M6 weights .ckpt file, in cell 7 set `--test_iters` to the number of iterations from the saved M6 weights (in this case 4000). Please pay attention that M6 saved model (ResNet18UNet_4000iter.ckpt) file is the only file with that name in the `save` folder.  Set `--mode` to `train_fine_tune`, pass the LRs `--lr_encoder`(we got: 2.7589e-05) and `--lr_decoder`(we got: 0.0012) and run cell 7. To test your model, change `get_patch` in cell 5 back to `augment=False` than run cell 5 again. Set `--mode` to `test` and run cell 7 (and cells 0-4, 6 if they weren’t run yet), please pay attention that your saved model .ckpt file is the only file with this name in the `save` folder. The results will be shown in the output of cell 7.

### Graphic Results
Open `Graphics_Results_Producing.ipynb`.
Run the first cell, 3 new "Models" folders will be made – M1, M2, M3. For every model to be examined, its training metrics that exist and were saved during the training (valid_loss.npy, valid_psnr_delta.npy, valid_ssim_delta.npy, valid_rmse_delta.npy, epoch_train_loss.npy, epoch_lrs.npy) should be manually inserted to some Model folder. Then, for every plot function that users wish to check, they should call it with the right Model(s) folder(s), and the right model(s) name(s) (all the models names are in a code cell under all the functions.
If users want to visualize the net's implementation, they can copy its class to the notebook, then edit the last cell with the name of the class and run it.

---
## 8. Acknowledgements
* This project adapts components and architecture from the RED-CNN PyTorch implementation by SSinyu.

## 9. License
* This project is licensed under the MIT License - see the LICENSE file for details.

## 10. References
1. Chen, Hu, et al. "Low-dose CT with a residual encoder-decoder convolutional neural network." IEEE transactions on medical imaging 36.12 (2017): 2524-2535.
2. Moen, Taylor R et al. “Low-dose CT image and projection dataset.” Medical physics vol. 48,2 (2021): 902-911. doi:10.1002/mp.14594
3. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016
4. In any publications that use these training datasets, please acknowledge the AAPM CT-MAR challenge data using the following sentences and references: This work used the datasets by the AAPM CT-MAR Grand Challenge [A, B]. The datasets were generated using a hybrid data simulation framework [C], which incorporates publicly available clinical images [D, E] and metal objects using the CatSim simulator in the open-source toolkit XCIST [F]
[A] E. Haneda, N. Peters, J. Zhang, G. Karageorgos, W. Xia, H. Paganetti, G. Wang, Y. Guo, J. Ma, H.S. Park, K. Jeon, F. Fan, M. Thies, and B. De Man, “AAPM CT Metal Artifact Reduction Grand Challenge,” Med. Phys., vol. 52, no. 10, e70050, 2025. https://doi.org/10.1002/mp.70050
[B] AAPM CT Metal Artifact Reduction (CT-MAR) Grand Challenge Benchmark Tool: https://github.com/xcist/example/tree/main/AAPM_datachallenge/
[C] N. Peters, E. Haneda, J. Zhang, G. Karageorgos, W. Xia, J. Verburg, G. Wang, H. Paganetti, and B. De Man, “A hybrid training database and evaluation benchmark for assessing metal artifact reduction methods for X-ray CT imaging,” Med. Phys., vol. 52, no. 10, e70020, 2025. https://doi.org/10.1002/mp.70020
[D] Yan K, Wang X, Lu L, Summers RM, "DeepLesion: Automated Mining of Large-Scale Lesion Annotations and Universal Lesion Detection with Deep Learning", Journal of Medical Imaging 5(3), 036501 (2018), doi: 10.1117/1.JMI.5.3.036501
[E] Goren, Nir, Dowrick, Thomas, Avery, James, & Holder, David. (2017). UCLH Stroke EIT Dataset - Radiology Data (CT). Zenodo. https://doi.org/10.5281/zenodo.1215676
[F] M. Wu, P. FitzGerald, J. Zhang, W.P. Segars, H. Yu, Y. Xu, B. De Man, "XCIST - an open access x-ray/CT simulation toolkit," Phys Med Biol. 2022 Sep 28;67(19). https://doi.org/10.1088/1361-6560/ac9174
5. Git implementation of RED-CNN: [SSinyu/RED-CNN](https://github.com/SSinyu/RED-CNN/blob/master/README.md).
