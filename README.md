# Image Metrics Calculator

This is the "offline" image metrics calculator for [cite our paper]. While the [online version](http://172.178.93.86:8000/) only allows the user to upload one image at a time, this offline version allows for batch image processing. It also allows the user to select the "context" (see our paper).

## How to Use

- Clone this repo: `git clone https://github.com/image-metrics-calculator/image-metrics-calculator.git`

- Enter the root directory: `cd image-metrics-calculator`

- (Important!) Download the checkpoints and other config files from [this link](https://1drv.ms/f/s!AqUu9ylMgqcDgu_kCVDD2xSiTAmHcw0?e=oOPkM5). Put the downloaded folder (`V3Det`) under `data`. The directory tree should look like:

    ```
    root/
    ├─ data/
    │  ├─ V3Det/
    │  │  ├─ annotations/
    │  │  ├─ checkpoints/
    ```

- Run `application.py` as instructed below.

Usage: 

```bash
python application.py 
    [-h] [--prob_threshold PROB_THRESHOLD] 
    [--context CONTEXT] [--device DEVICE] [--img_dir IMG_DIR] 
    [--out_dir OUT_DIR]
```

- `-h`, `--help`: show this help message and exit
- `--prob_threshold`: Threshold for the object confidence in the user-uploaded image. Any objects with confidence below this threshold will be removed. Default 0.1.
- `--context`: (default "user") "general" will use the V3D training data as the context; "user" will use the user-uploaded image as the context. Default "user."
- `--device`: The device to use for inference. "cpu" for CPU and "cuda:0" for GPU.
- `--img_dir`: The directory to save the uploaded images.
- `--out_dir`: The directory to save the outputs. Default root directory.                  


<!-- ## Author Team
- **Amrita Dey**, Department of Marketing, Denver University
- Tianyu Gu, Department of Marketing, University of Utah
- Yu Zhu, Department of Operations and Information Systems, University of Utah
- Steve J. Carson, Department of Marketing, University of Utah -->
