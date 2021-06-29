# References:
**Projected adapted directly from:**
**Deep Neural Networks Improve Radiologists' Performance in Breast Cancer Screening**\
Nan Wu, Jason Phang, Jungkyu Park, Yiqiu Shen, Zhe Huang, Masha Zorin, Stanisław Jastrzębski, Thibault Févry, Joe Katsnelson, Eric Kim, Stacey Wolfson, Ujas Parikh, Sushma Gaddam, Leng Leng Young Lin, Kara Ho, Joshua D. Weinstein, Beatriu Reig, Yiming Gao, Hildegard Toth, Kristine Pysarenko, Alana Lewin, Jiyon Lee, Krystal Airola, Eralda Mema, Stephanie Chung, Esther Hwang, Naziya Samreen, S. Gene Kim, Laura Heacock, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras\
IEEE Transactions on Medical Imaging\
2019

Github: https://github.com/nyukat/breast_cancer_classifier
Arxiv: https://arxiv.org/pdf/1903.08297v1.pdf

**Dataset model was adapted on:**
[Ref paper/Mini-DDSM] C.D. Lekamlage, F. Afzal, E. Westerberg and A. Cheddad, “Mini-DDSM: Mammography-based Automatic Age Estimation,” in the 3rd International Conference on Digital Medicine and Image Processing (DMIP 2020), ACM, Kyoto, Japan, November 06-09, 2020, pp: 1-6.
And
[Ref DDSM] Michael Heath, Kevin Bowyer, Daniel Kopans, Richard Moore and W. Philip Kegelmeyer, in Proceedings of the Fifth International Workshop on Digital Mammography, M.J. Yaffe, ed., 212-218, Medical Physics Publishing, 2001. ISBN 1-930524-00-5.

Kaggle: https://www.kaggle.com/cheddad/miniddsm2

# Team
Kyle Murray, kpmurray@bu.edu
Vella(Yiting) Liu, vellaliu@bu.edu

# Slides
https://drive.google.com/file/d/1KuzzHebIlnW2rH1KJo9FSw_4mN6jAK9k/view?usp=sharing

# Demo (taken from original README):
**For more in depth information regarding the model and running, please see the original README.md file within the readme directory.**
### Exam-level

Here we describe how to get predictions from *view-wise* model, which is our best-performing model. This model takes 4 images from each view as input and outputs predictions for each exam.

```bash
bash run.sh
``` 
will automatically run the entire pipeline and save the prediction results in csv. 

We recommend running the code with a gpu (set by default). To run the code with cpu only, please change `DEVICE_TYPE` in `run.sh` to 'cpu'.  

If running the individual Python scripts, please include the path to this repository in your `PYTHONPATH` . 

You should obtain the following outputs for the sample exams provided in the repository. 

Predictions using *image-only* model (found in `sample_output/image_predictions.csv` by default):

| index | left_benign | right_benign | left_malignant | right_malignant |
| ----- | ----------- | ------------ | -------------- | --------------- |
| 0     | 0.0580      | 0.0754       | 0.0091         | 0.0179          |
| 1     | 0.0646      | 0.9536       | 0.0012         | 0.7258          |
| 2     | 0.4388      | 0.3526       | 0.2325         | 0.1061          |
| 3     | 0.3765      | 0.6483       | 0.0909         | 0.2579          |


Predictions using *image-and-heatmaps* model (found in `sample_output/imageheatmap_predictions.csv` by default):

| index | left_benign  | right_benign | left_malignant | right_malignant |
| ----- | ------------ | ------------ | -------------- | --------------- |
| 0     | 0.0612       | 0.0555       | 0.0099         | 0.0063          |
| 1     | 0.0507       | 0.8025       | 0.0009         | 0.9000          |
| 2     | 0.2877       | 0.2286       | 0.2524         | 0.0461          |
| 3     | 0.4181       | 0.3172       | 0.3174         | 0.0485          |

### Single Image

Here we also upload *image-wise* model, which is different from and performs worse than the *view-wise* model described above. The csv output from *view-wise* model will be different from that of *image-wise* model in this section. Because this model has the benefit of creating predictions for each image separately, we make this model public to facilitate transfer learning.

To use the *image-wise* model, run a command such as the following:

```bash
bash run_single.sh "sample_data/images/0_L_CC.png" "L-CC"
``` 

where the first argument is path to a mammogram image, and the second argument is the view corresponding to that image.

You should obtain the following output based on the above example command:

```
Stage 1: Crop Mammograms
Stage 2: Extract Centers
Stage 3: Generate Heatmaps
Stage 4a: Run Classifier (Image)
{"benign": 0.040191903710365295, "malignant": 0.008045293390750885}
Stage 4b: Run Classifier (Image+Heatmaps)
{"benign": 0.052365876734256744, "malignant": 0.005510155577212572}
```

#### Image-level Notebook

We have included a [sample notebook](sample_notebook.ipynb) that contains code for running the classifiers with and without heatmaps (excludes preprocessing).

## Data

To use one of the pretrained models, the input is required to consist of at least four images, at least one for each view (L-CC, L-MLO, R-CC, R-MLO). 

The original 12-bit mammograms are saved as rescaled 16-bit images to preserve the granularity of the pixel intensities, while still being correctly displayed in image viewers.

`sample_data/exam_list_before_cropping.pkl` contains a list of exam information before preprocessing. Each exam is represented as a dictionary with the following format:

```python
{
  'horizontal_flip': 'NO',
  'L-CC': ['0_L_CC'],
  'R-CC': ['0_R_CC'],
  'L-MLO': ['0_L_MLO'],
  'R-MLO': ['0_R_MLO'],
}
```

We expect images from `L-CC` and `L-MLO` views to be facing right direction, and images from `R-CC` and `R-MLO` views are facing left direction. `horizontal_flip` indicates whether all images in the exam are flipped horizontally from expected. Values for `L-CC`, `R-CC`, `L-MLO`, and `R-MLO` are list of image filenames without extension and directory name. 

Additional information for each image gets included as a dictionary. Such dictionary has all 4 views as keys, and the values are the additional information for the corresponding key. For example, `window_location`, which indicates the top, bottom, left and right edges of cropping window, is a dictionary that has 4 keys and has 4 lists as values which contain the corresponding information for the images. Additionally, `rightmost_pixels`, `bottommost_pixels`, `distance_from_starting_side` and `best_center` are added after preprocessing. 
Description for these attributes can be found in the preprocessing section. 
The following is an example of exam information after cropping and extracting optimal centers:

```python
{
  'horizontal_flip': 'NO',
  'L-CC': ['0_L_CC'],
  'R-CC': ['0_R_CC'],
  'L-MLO': ['0_L_MLO'],
  'R-MLO': ['0_R_MLO'],
  'window_location': {
    'L-CC': [(353, 4009, 0, 2440)],
    'R-CC': [(71, 3771, 952, 3328)],
    'L-MLO': [(0, 3818, 0, 2607)],
    'R-MLO': [(0, 3724, 848, 3328)]
   },
  'rightmost_points': {
    'L-CC': [((1879, 1958), 2389)],
    'R-CC': [((2207, 2287), 2326)],
    'L-MLO': [((2493, 2548), 2556)],
    'R-MLO': [((2492, 2523), 2430)]
   },
  'bottommost_points': {
    'L-CC': [(3605, (100, 100))],
    'R-CC': [(3649, (101, 106))],
    'L-MLO': [(3767, (1456, 1524))],
    'R-MLO': [(3673, (1164, 1184))]
   },
  'distance_from_starting_side': {
    'L-CC': [0],
    'R-CC': [0],
    'L-MLO': [0],
    'R-MLO': [0]
   },
  'best_center': {
    'L-CC': [(1850, 1417)],
    'R-CC': [(2173, 1354)],
    'L-MLO': [(2279, 1681)],
    'R-MLO': [(2185, 1555)]
   }
}
```

The labels for the included exams are as follows:

| index | left_benign | right_benign | left_malignant | right_malignant |
| ----- | ----------- | ------------ | -------------- | --------------- |
| 0     | 0           | 0            | 0              | 0               |
| 1     | 0           | 0            | 0              | 1               |
| 2     | 1           | 0            | 0              | 0               |
| 3     | 1           | 1            | 1              | 1               |

### Additional Running Notes
* The dataset this model was extended to is not present in this repo due to space constraints. Please see "/projectnb/cs523/kpmurray/project/breast_cancer_classifier/sample_data/new_images" for the data the model was adapted on
* The fourImage.qsub and singleImage.qsub files will submit jobs to the SCC for the main model and single image variant accordingly.
