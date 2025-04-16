# NYCU Computer Vision 2025 Spring HW2

**StudentID:** 111550135 
**Name:** 林李奕

---

## Introduction

This repository contains the solution for HW2 of NYCU Computer Vision 2025 Spring.  
We tackle **digit detection** and **digit sequence recognition** on a custom COCO‐formatted dataset using:

- A **Faster R‑CNN ResNet‑50 FPN v2** backbone (pretrained on COCO).  
- A **custom anchor generator** for small digits.  
- **Soft‑NMS** post‑processing to suppress overlapping detections.  
- A simple left‑to‑right ordering of detected boxes to form digit sequences.

---

## How to install

1. Clone this repository:

    ```bash
    git clone https://github.com/owo0505/NYCU-Computer-Vision-2025-Spring-HW2.git
    cd NYCU-Computer-Vision-2025-Spring-HW2
    ```

2. Install dependencies (make sure you have Python ≥ 3.8):

    ```bash
    pip install -r requirements.txt
    ```

3. Prepare your dataset in this format:

    ```
    nycu-hw2-data/
    ├─ train/
    │   ├─ 1.png
    │   ├─ 2.png
    │   ├─ ...
    │      
    ├─ train.json
    │   
    ├─ valid/
    │   ├─ 1.png
    │   ├─ 2.png
    │   ├─ ...
    │
    ├─ valid.json
    │   
    └─ test/
        ├─ 1.png
        ├─ 2.png
        ├─ ...

    ```

4. Train the model:

    ```bash
    # Make sure data_dir in train.py is set to your actual dataset path
    python train.py
    ```

5. Run inference (after training completes and best models are saved):

    ```bash
    # Remember to change test_dir in inference.py to the actual test dataset path
    python inference.py
    ```

---

## Performance snapshot

| Model             | Tets mAP | Test Accuracy                                       |
|------------------|---------------------|---------------------------------------------|
| Base Anchor      | ~38.7%              | ~82.0%  |
| Custom Anchor    | ~40.0%              | ~84.0%    |

![leaderboard snapshot](results/snapshot.png)

You can find training curves and mAP curve output under Result:

- `learning_curve.png`
- `mAP_curve.png`

---
