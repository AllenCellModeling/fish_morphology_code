# 1 - Local structure organization

## Training of CNN-based classififier for EGFP-alpha-actinin-2 patterns

The folder `local_organization` contains the data and code to train the ResNet18 convolutional neural network used to classify images of EGFP-alpha-actinin-2 into six classes:

* Background
* Diffuse/others
* Fibers
* Disorganized puncta
* Organized Puncta
* Organized Z-disks

### Requirerments to train and use a model

aicssegmentation==0.1.12
Pillow==6.2.0
torch==1.3.0
torchvision==0.4.1

Be aware that both training and inference process can be considerably slow if you don't have GPU available.

### How to train a new model

A new model can be trained with

```
cd local_organization/train/
python model_training.py
```

This will train a new model for 750 epochs. At each epoch one instance of the model will be saved as a `.pth` file in the folder `local_organization/train/models/`.

### Model selection

As the model is training, training and validation losses and accuracies are saved in the folder `local_organization/train/` as numpy arrays with name `CNN_epoch_training_scores.npy`, `CNN_epoch_test_score.npy`, `CNN_epoch_training_losses.npy` and `CNN_epoch_test_loss.npy`, respectively. By inspecting the values in these arrays one can select the model with best performance.

The `.pth` file corresponding to the best model should be manually copied to the folder `local_organization/best_model/` for further use.

### Inference on the data used in the paper

The script `inference/inference.py` uses the best model selected by the user to perform classification on new images. In addition, this script also uses the Allen Cell Structure Segmenter [1] to identify background regions in the input data and mask them out from the final classification maps. The input data is organized in the CSV file `database/database.csv`. Each row in this file corresponds to an input z-stack for which the inference will be performed on. Results are saved in the folder `structure_organization/output/` as the z-stacks are processed one at the time. An additional CSV file `database/database_cell.csv` is used to load the single cell manual segmentation. For each input z-stack an output z-stack is produced. The slices of the output z-stack are:

1. Highest mean intensity slice form the original input z-stack.
2. Probaility map for class Diffuse/others
3. Probaility map for class Fibers
4. Probaility map for class Disorganized puncta
5. Probaility map for class Organized Puncta
6. Probaility map for class Organized Z-disks
7. Final classification map based on the highest probability
8. Single cell manual segmentation 

The classes in the 7th slice are encoded as follow:

0. Background
1. Diffuse/others
2. Fibers
3. Disorganized puncta
4. Organized Puncta
5. Organized Z-disks

** Disclaimer on intensity issues.

# 2 - Global structural alignment

We implemented the method describe in [2] to quantify the global alignment of EGFP-alpha-actinin-2 patterns.

## How to run the calculation for a particular FOV

The folder `global_alignment` contains the script `alignment.py` that is used to compute the global alignment metrics for aall cells in a given FOV. This scripts uses the CSVs files `database/database.csv` and `database/database_cell.csv` to relate full z-stack with the single segmentation. To run the script on FOV 0, one can do

```
python alignment.py --fov 0
```

This command will produce a CSV file with name `output/fov_0.csv`. Each row of this CSV file corresponds to one cell from FOV 0 and columns are the diferent metrics of global structural alignment used in the paper:

* Maximum_Coefficient_Variation
* Peak_Angle
* Peak_Distance
* Peak_Height

Other files in the folder `global_alignment` can be used to run multiple FOVs at once in a cluster.

# 3 - Merging local organization and global alignment features for a single FOV

We use the script `features.py` found in folder `tools/` to compute local structure organization metrics for a given FOV based on the pattern classification done in step (1). To run the script for FOV 0, one should do

```
python features.py --fov 0
```

This script will output the following metrics:

* FOVId
* CellId
* Total_Area
* Frac_Area_Background
* Frac_Area_DiffuseOthers
* Frac_Area_Fibers
* Frac_Area_Disorganized_Puncta
* Frac_Area_Organized_Puncta
* Frac_Area_Organized_ZDisks
* Prob_DiffuseOthers
* Prob_Fibers
* Prob_Disorganized_Puncta
* Prob_Organized_Puncta
* Prob_Organized_ZDisks
* Intensity_Median
* Intensity_Integrated
* Intensity_Median_BackSub
* Intensity_Integrated_BackSub
* Background_Value

This metrics are appended as new columns in the CSV file `output/fov_0.csv`.

# 4 - Merging information of all FOVS in a single CSV file

After the features have been computed for all FOVs, we run the script `merge.py` found in folder `tools/` to merge all the feature in a singel CSV file that is then used for making plots.

# 5 - Extra figures

Jupyter notebooks can be found in the folder `figs/` to help reproduce some of the figures used in the paper

# References

[1] - Chen J, Ding L, Viana MP, Hendershott MC, Yang R, Mueller IA, Rafelski SM. The Allen Cell Structure Segmenter: a new open source toolkit for segmenting 3D intracellular structures in fluorescence microscopy images. bioRxiv. 2018 Jan 1:491035.

[2] - Sutcliffe, M.D., Tan, P.M., Fernandez-Perez, A., Nam, Y.J., Munshi, N.V. and Saucerman, J.J., 2018. High content analysis identifies unique morphological features of reprogrammed cardiomyocytes. Scientific reports, 8(1), pp.1-11.


















