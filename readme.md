# $\text{SAM-MSF}^2$

$\text{SAM-MSF}^2$ is a fine-tuning model based on SAMMed2d, mainly designed for vascular segmentation.

## Installation Steps

You can follow the steps below to install:

1. Clone the project to your local machine using the following command:
   ```
   git clone https://github.com/YuancongLiang/SAM-MSF2.git
   ```

2. Navigate to the project directory:
   ```
   cd SAM-MSF2
   ```

3. Create and activate a virtual environment (optional but recommended):
   ```
   conda create -n sammsf2 python=3.9
   ```

4. Install all the required dependencies:
   ```
   pip install -r requirements.txt
   pip install surface-distance/
   ```

## Usage

Here are some common usage examples:
Before that, you need to download the FIVES dataset, which is a high-quality dataset for retinal vessel segmentation.
```
wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/34969398/FIVESAFundusImageDatasetforAIbasedVesselSegmentation.rar?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIYCQYOYV5JSSROOA/20240619/eu-west-1/s3/aws4_request&X-Amz-Date=20240619T120301Z&X-Amz-Expires=10&X-Amz-SignedHeaders=host&X-Amz-Signature=85f6727c5ddf36ad84c7ae51c69d3341b945d87a1b22e0bb562f77399d74b8aa
```
Or you can download it from https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1
Then, you need a pre training weight from SAMMed2d or SAM-vit-b
SAMMed2d:
download from https://drive.google.com/file/d/1ARiB5RkSsWmAB_8mqWnwDF8ZKTtFwsjl/view?usp=drive_link
sam-vit-b:
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
or download from https://github.com/facebookresearch/segment-anything

- Example 1: Run the training script
  ```
  python train_without_prompt.py --run_name fives --epochs 60 --batch_size 32 --resume pretrain_model/sam-med2d_b.pth
  ```

- Example 2: Perform inference using a pre-trained model
  ```
  python predict.py --sam_checkpoint ./workdir/models/fives/epoch60_sam.pth
  ```

Feel free to modify and adjust these examples according to your specific task and requirements.

## Contributing Guidelines

If you would like to contribute to this project, please follow these steps:

1. Fork the project and make your modifications.

2. Submit a Pull Request to submit your changes to our repository.

3. We will review your Pull Request and merge appropriate changes.

## Copyright and License

This project is licensed under the Apache License. For more details, please refer to the [LICENSE](LICENSE).
