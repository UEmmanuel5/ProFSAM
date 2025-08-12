# ProFSAM – Promptable Fire Segmentation using SAM2

This repository contains **only the ProFSAM script folders and supporting code** for evaluating **SAM2**, **MobileSAM**, and **TinySAM** variants on fire segmentation tasks used in the study **"Promptable Fire Segmentation: Unleashing SAM2’s Potential for Real-Time Mobile Deployment with Strategic Bounding Box Guidance"**.
It provides a framework for evaluating **SAM2**, **MobileSAM**, and **TinySAM** variants on fire segmentation tasks with multiple prompting strategies, as well as YOLOv11-based fire detection.

**Important**  
To run the notebooks, you must **first clone/download the official SAM2.1, MobileSAM, and TinySAM repositories** from their respective sources, then place the provided `script/` folders from this repo into their correct locations.


---

##  Quick Start


1. **Clone the Repo (ProFSAM scripts)**
   ```bash
   git clone https://github.com/UEmmanuel5/ProFSAM.git
   cd ProFSAM
   ````

2. **Clone model repositories**

   * **SAM2.1** (Facebook Research) → [https://github.com/facebookresearch/sam2.1](https://github.com/facebookresearch/sam2/tree/sam2.1)
   * **MobileSAM** (Chaoning Zhang) → [https://github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
   * **TinySAM** (Xinghao Chen) → [https://github.com/xinghaochen/TinySAM](https://github.com/xinghaochen/TinySAM)

3. **Place the script folders**
   * Copy the `script/` folder from this ProFSAM repo into each cloned model repo:

   ```
   ProFSAM/SAM2.1/script/       →   <path-to-SAM2.1-repo>/script/
   ProFSAM/MobileSAM/script/  →   <path-to-MobileSAM-repo>/script/
   ProFSAM/TinySAM/script/    →   <path-to-TinySAM-repo>/script/
   ```


4. **Install dependencies**
    ```bash
    # First, install the correct GPU-enabled PyTorch version for your system:
    #   Check: https://pytorch.org/get-started/locally/
    # Example (CUDA 12.6):
    #   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    #
    # Then install the remaining dependencies:
    pip install -r requirements.txt
    #
    # Note: requirements.txt contains only the core dependencies.
    #       Depending on your environment, you may need to manually install extra packages
    #       if prompted when running the notebooks.
    ```


5. **Run notebooks**
   * Open Jupyter and run the desired script in the corresponding model repo.



---

##  Repository Structure

```
## Repository Structure

Below is the expected local directory organization after you have:

- Cloned this ProFSAM repo
- Downloaded/cloned the SAM2.1, MobileSAM, and TinySAM repositories
- Placed the provided `script/` folders into their respective model repos
- Downloaded the required datasets and model weights or checkpoints and configs

Note: This layout is for your local PC setup. Large model weights and datasets are not included in this GitHub repository.

ProFSAM/
│
├── SAM2.1/                        # SAM2.1 official repo (with added scripts)
│   ├── checkpoints/               # Pretrained SAM2.1 model weights
│   ├── sam2/configs/              # Config files for SAM2.1 variants
│   └── script/                    # Evaluation scripts
│       ├── images/                # Notebooks for image-based evaluation
│       └── videos/                # Notebooks for video-based evaluation
│
├── MobileSAM/                     # MobileSAM official repo (with added scripts)
│   ├── weights/                   # MobileSAM pretrained model weights (e.g., mobile_sam.pt)
│   └── script/                    # Evaluation scripts (images & videos)
│
├── TinySAM/                       # TinySAM official repo (with added scripts)
│   ├── weights/                   # TinySAM pretrained model weights (e.g., tinysam_42.3.pth)
│   └── script/                    # Evaluation scripts (images & videos)
│
├── YOLO/
│   └── Fire_best.pt               # YOLOv11 trained fire detection weights
│
├── requirements.txt               # Core dependencies for running notebooks/scripts
│
├── assets/                        # Diagrams, figures, small static images used in README/notebooks
│   ├── SAM_architecture.drawio.svg
│   └── B-box.drawio.svg
│
└── dataset/                       # (Placeholders) Structure for downloaded datasets
    ├── khan/                      # Khan Fire Segmentation Dataset
    │   └── images/                # [All Khan dataset images go here]
    │
    ├── roboflow/                  # Roboflow Fire Dataset
    │   └── images/                # [All Roboflow dataset images go here]
    │
    └── foggia/                    # Foggia Video Dataset
        └── videos/                # [All 5 Foggia dataset videos go here]

```

---

##  Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/UEmmanuel5/ProFSAM.git
   cd ProFSAM
   ```

2. **Install dependencies** (Python ≥3.9 recommended)

   ```bash
   pip install -r requirements.txt
    # Note: requirements.txt contains only the core dependencies.
    # Depending on your environment, you may need to manually install extra packages
    # if prompted when running the notebooks.
   ```

3. **Download SAM2.1 checkpoints & configs**
   Place them under:

   ```
   SAM2/checkpoints/sam2.1_hiera_x/       # Replace x with checkpoint of variant
   SAM2/sam2/configs/sam2.1/sam2.1_hiera_x/       # Replace x with config of variant
   ```

   If newer versions exist in the [official SAM2 repo](https://github.com/facebookresearch/sam2/tree/sam2.1), download them directly.

4. **Verify YOLO weights**
   Ensure `./YOLO/Fire_best.pt` is present. This is the trained YOLOv11n detector for generating bounding box prompts.

5. **Create output directory**

   ```bash
   mkdir outputs
   ```

---

##  Prompt Types

| Code     | Description                                                                   |
| -------- | ----------------------------------------------------------------------------- |
| `Auto`   | SAM2 automatic segmentation with no external prompts (baseline).              |
| `SP`     | Single positive point at the center of detected bounding box.                 |
| `SP+SN`  | Positive center point + one negative point outside all detected boxes.        |
| `MP`     | Multiple positive points in a 3×3 grid inside the box, filtered by HSV color. |
| `Box`    | Bounding box prompt directly from YOLO detection.                             |
| `Box+SP` | Bounding box prompt + center point.                                           |
| `Box+MP` | Bounding box prompt + multiple positive points (MP).                          |

---


##  Datasets

The experiments in this work use the following publicly available datasets:

| Dataset                            | Description                                                                                    | Link                                                                                                                                                                                           |
| ---------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Khan Fire Segmentation Dataset** | 600 pixel-level annotated images from YouTube videos, covering various outdoor fire scenarios. | [Dataset Link](https://github.com/hayatkhan8660-maker/Fire_Seg_Dataset?tab=readme-ov-file)                                                                                                                 |
| **Roboflow Fire Dataset**          | 7,040 annotated images from multiple sources (indoor and outdoor fire scenarios).              | [Dataset Link](https://universe.roboflow.com/firesegpart1/fire-seg-part1/dataset/21)                                                                                                           |
| **Foggia Video Dataset**           | Five annotated video sequences of fire incidents for temporal stability evaluation.            | [Dataset Link](https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/) |
| **FASDD**                          | Large-scale flame detection dataset with over 100,000 images (only 51,749 were used to train the YOLOv11n detector used in this paper).      | [Dataset Link](https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda)                                                                                                                                          |

---

##  Fire Segmentation Pipeline

![Fire Segmentation Pipeline](assets/SAM_architecture.drawio.svg)

**Figure:** The ProFSAM pipeline.

1. Input image or video frame is processed by **YOLOv11n** to detect fire regions.
2. Detected bounding boxes are passed as prompts (with optional points) to **SAM2 / MobileSAM / TinySAM**.
3. The model outputs a segmentation mask, which is overlaid on the original frame for visualization.


![Fire Segmentation Pipeline](assets/B-box.drawio.svg)

**Figure:** Masks gotten through this pipeline vs Groundtruth masks.

---

##  Notebooks Reference

| Model         | Data Type | Prompt Type | Notebook Path                                                                                  |
| ------------- | --------- | ----------- | ---------------------------------------------------------------------------------------------- |
| **SAM2.1 (All)**    | Image     | Auto        | [SAM2/script/images/auto.ipynb](SAM2/script/images/auto.ipynb)                                 |
| **SAM2.1 (All)**    | Image     | Box         | [SAM2/script/images/b-box.ipynb](SAM2/script/images/b-box.ipynb)                               |
| **SAM2.1 (All)**    | Image     | Box+MP      | [SAM2/script/images/b-box+mp.ipynb](SAM2/script/images/b-box+mp.ipynb)                         |
| **SAM2.1 (All)**    | Image     | Box+SP      | [SAM2/script/images/b-box+sp.ipynb](SAM2/script/images/b-box+sp.ipynb)                         |
| **SAM2.1 (All)**    | Image     | MP          | [SAM2/script/images/mp.ipynb](SAM2/script/images/mp.ipynb)                                     |
| **SAM2.1 (All)**    | Image     | SP          | [SAM2/script/images/sp.ipynb](SAM2/script/images/sp.ipynb)                                     |
| **SAM2.1 (All)**    | Image     | SP+SN       | [SAM2/script/images/sp+sn.ipynb](SAM2/script/images/sp+sn.ipynb)                               |
| **SAM2.1 (Large)**    | Video     | Box         | [SAM2/script/videos/b-box.ipynb](SAM2/script/videos/b-box.ipynb)                               |
| **SAM2.1 (Base_Plus)**    | Video     | Box+MP      | [SAM2/script/videos/b-box+mb.ipynb](SAM2/script/videos/b-box+mb.ipynb)                         |
| **MobileSAM (mobile_sam.pt)** | Image     | Box         | [MobileSAM/script/images/b-box+mobilesam.ipynb](MobileSAM/script/images/b-box+mobilesam.ipynb) |
| **MobileSAM (mobile_sam.pt)** | Video     | Box         | [MobileSAM/script/videos/b-box+mobilesam.ipynb](MobileSAM/script/videos/b-box+mobilesam.ipynb) |
| **TinySAM (tinysam_42.3.pth)**   | Image     | Box         | [TinySAM/script/images/b-box+tinysam.ipynb](TinySAM/script/images/b-box+tinysam.ipynb)         |
| **TinySAM (tinysam_42.3.pth)**   | Video     | Box         | [TinySAM/script/videos/b-box+tinysam.ipynb](TinySAM/script/videos/b-box+tinysam.ipynb)         |

---

##  Citation

If you use this repository in your research, please cite:

```bibtex
@article{ugwu2025promptablefiresam,
  title={Promptable Fire Segmentation: Unleashing SAM2’s Potential for Real-Time Mobile Deployment with Strategic Bounding Box Guidance},
  author={Ugwu, Emmanuel U. and Zhang, Xinming},
  year={2025}
}
