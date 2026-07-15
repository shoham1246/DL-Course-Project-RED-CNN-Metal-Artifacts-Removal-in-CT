# DL-Course-Project-RED-CNN-Metal-Artifacts-Removal-in-CT


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
## Acknowledgements
* This project adapts components and architecture from the [RED-CNN PyTorch implementation](https://github.com/SSinyu/RED-CNN) by SSinyu.
