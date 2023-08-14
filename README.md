# Liver Lesions Detection based on Ultrasound Image Hackathon

This repository presents the 1st place solution for the Liver Lesions Detection based on Ultrasound Image Hackathon
held from August 12th to 14th, 2023, at Mahidol University.

The solution leverages the [DINO architecture](https://arxiv.org/abs/2203.03605) with a Swin Transformer backbone to detect
three classes of liver lesions (cystic, fibrosis, solid) as well as normal liver images. Our solution achieved a mAP70 score of 0.5037 on the public
leaderboard (using 25% of the test data) and 0.5263 on the private leaderboard (utilizing the entire test data).

## Our Observations

- The Transformer backbone outperforms the convolutional backbone. We attribute this to the inherent noise present in ultrasound images. The Transformer's backbone is more adept at learning improved ultrasound image representations.
- Incorporating all images during training is crucial. In the `mmdetection` dataloader configuration, the option `filter_cfg=dict(filter_empty_gt=False)` is utilized.  This allows the model to see more negative examples during training.

## Model Configurations

For detailed training configuration, refer to the file `dino-5scale_swin-l_8xb2-12e_liver.py`. Additional information can be found in the [`mmdetection`](https://mmdetection.readthedocs.io/en/latest/user_guides/config.html) documentation.

## Members

- Zaw Htet Aung
- Kittinan Srithaworn
- Titipat Achakulvisut
