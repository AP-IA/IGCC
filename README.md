# IGCC: Importance-Guided Cross-Image Communication for Fine-Grained Recognition

This repository provides the code of **IGCC**, a fine-grained visual classification framework that enables cross-image communication among discriminative regions using importance-aligned patch graphs.

The framework is built upon a Dynamic Swin Transformer backbone and is evaluated on multiple fine-grained visual classification (FGVC) benchmarks, including CUB-200-2011, Stanford Cars, Stanford Dogs, FGVC-Aircraft, NABirds, and iNaturalist 2017.

---

###  Environment Setup

We recommend using **conda** to create an isolated environment.

```bash
conda create -n igcc python=3.10.12 -y
conda activate igcc
pip install -r requirements.txt
````

---

###  Training

Example training command on **CUB-200-2011**:

```bash
python ./train.py \
  --name cub224 \
  --config ./configs/CUB200.yaml \
  --sample_classes 2 \
  --sample_images 10 \
  --arch swin-base \
  --img_size 224 \
  --gpus 0
```

---

###  Project Structure

```text
.
├── configs
│   ├── Cars.yaml
│   ├── CUB200.yaml
│   ├── Dogs.yaml
│   ├── FGVC-Aircraft.yaml
│   ├── iNat17.yaml
│   └── NABirds.yaml
│
├── data
│   ├── aircraft
│   │   ├── classes.txt
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── data
│   ├── car
│   ├── cub
│   ├── inat17
│   ├── dog
│   └── nabird
│
├── modules
│   ├── __init__.py
│   ├── cbam.py
│   ├── common.py
│   ├── datasets.py
│   ├── losses.py
│   └── utils.py
│
├── network
│   ├── DSwin.py
│   ├── GlobalBranch.py
│   ├── PFE.py
│   └── PGA.py
│
├── requirements.txt
├── train.py
└── README.md
```

---

###  Datasets

This project supports the following fine-grained visual classification benchmarks.
Dataset split files (`train.txt`, `val.txt`, `all.txt`) are already provided under the `data/` directory.
You only need to download the raw images from the official sources.

**Official Download Links:**

* **CUB-200-2011**
  [http://www.vision.caltech.edu/datasets/cub_200_2011/](http://www.vision.caltech.edu/datasets/cub_200_2011/)

* **Stanford Cars**
  [https://ai.stanford.edu/~jkrause/cars/car_dataset.html](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

* **Stanford Dogs**
  [http://vision.stanford.edu/aditya86/ImageNetDogs/](http://vision.stanford.edu/aditya86/ImageNetDogs/)

* **FGVC-Aircraft**
  [https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)

* **NABirds**
  [https://github.com/visipedia/nabirds](https://github.com/visipedia/nabirds)

* **iNaturalist 2017**
  [https://github.com/visipedia/inat_comp](https://github.com/visipedia/inat_comp)

After downloading, please organize the image folders according to the paths specified in each dataset configuration file (`configs/*.yaml`).

---


