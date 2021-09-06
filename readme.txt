Models have been trained using Tensorflow v2.4. Additional libraries required are numpy, scipy, pandas, SimpleITK

Original Training images should be in 'TrainingImg' & 'TrainingMask' Folders. Similarly, Validation images should be put int 'ValidationImg' and the inference output will be produced in 'ValidationMask'

Simple commands to run each stage (pre-processing, localiser training, segmentation training, validation case inference) should be through the following execution:
python 00_split_flare_labels_to_nii.py
cd 01_Localiser
sh train_localiser_all_regions.sh
cd ../02_Segmenter
sh train_segmenter_all_regions.sh
cd ../03_Inference
python FLARE21_inference.py

Detailed instructions:
Pre-process by executing split_flare_labels_to_nii.py which splits each label into individual files and attempts to detect right/left kidney as contiguous regions
Processed .nii images should be in ct, rkid, lkid, liver, spleen, pancreas folders

Training runs for each organ independently with 2 stages, localisation and segmentation.
Both are adaptations of 3D unet at low- and high-resolution cropped to a region-adapted bounding box for segmentation

To train the first stage, it should be possible to run 'train_localiser_all_regions.sh' in the '01_Localiser' folder
This will execute the training script for each region of the form:
python Region_Finder_min_output.py liver "400 400 1500" "96 96 144" 25.0
python Region_Finder_min_output.py lt_kidney "400 400 1500" "96 96 144" 25.0
...
The arguments following the command are the region (corresponding to the input label folders), the physical extent of the whole body resampling space in mm (xyz convention), the resolution of the WB resampling space (xyz), and the radius of the label expansion used to counter potential class imbalance with small labels in a large physical volume. 25mm seems to work universally.

Similarly, the segmentation stage scripts are located in the '02_Segmenter' folder. Individual regions are trained with the form:
python Cropped_Segmenter_min_output.py liver "128 128 112"
Post script arguments are the region name and resolution of the cropped region for segmentation. The physical extents are taken from the accompanying .csv in the segmenter directory.

Commands to train segmentation for each region are located in 'train_segmenter_all_regions.sh'. Note: the resolution for each is slightly different to achieve bounding box with approximately cubic aspect ratio


Running inference on the validation cases with the trained models is completed in folder '03_Inference' which should run for all cases in the 'ValidationImg' folder with the command:
python FLARE21_inference.py

Output .nii.gz files will appear in the ValidationMask directory, resampled to the original CT input with voxels labelled 0-4 corresponding to background or one of the four tissue types.
