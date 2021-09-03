Models have been trained using Tensorflow v2.4. Additional libraries required are numpy, scipy, pandas, SimpleITK

Original Training images should be in 'TrainingImg' & 'TrainingMask' Folders. Similarly, Validation images should be put into 'ValidationImg' and the inference output will be produced in 'ValidationMask'
Pre-process by executing split_flare_labels_to_nii.py which splits each label into individual files and attempts to detect right/left kidney as contiguous regions based on the centre of mass in the CT image set.
Processed .nii images should be in ct, rkid, lkid, liver, spleen, pancreas folders

Training runs for each organ independently with 2 stages, localisation and segmentation.
Both are adaptations of 3D unet at low- and high-resolution cropped to a region-adapted bounding box for segmentation

To train the first stage, it should be possible to run 'train_localiser_all_regions.sh' in the '01_Localiser' folder
This will execute the training script for each region of the form:
python Region_Finder_min_output.py liver "400 400 1500" "96 96 144" 25.0
python Region_Finder_min_output.py lt_kidney "400 400 1500" "96 96 144" 25.0
...


Similarly, the segmentation stage scripts are located in the '02_Segmenter' folder. Individual regions are trained with the form:
python Cropped_Segmenter_min_output.py liver "128 128 112"

Commands to train segmentation for each region are located in 'train_segmenter_all_regions.sh'. Note: the resolution for each is slightly different to achieve bounding box with approximately cubic aspect ratio


Running inference on the validation cases with the trained models is completed in folder '03_Inference'


