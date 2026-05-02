<h2>TensorFlow-FlexUNet-Image-Segmentation-Pardoe-Hippocampus-Amygdala (2026/05/01)</h2>

Sarah T. Arai<br>
Software Laboratory antillia.com<br>
<br>
This is the first experiment of Image Segmentation for Pardoe-Hippocampus-Amygdala,
 based on our 
TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and a 512x600 pixels upscaled PNG 
<a href="https://drive.google.com/file/d/1jR8leWcaQ5nJ_mFvCxgUWBRCU79alWXo/view?usp=sharing">
Pardoe-Hippocampus-Amygdala-ImageMask-Dataset.zip
</a> with colorized masks, 
which was derived by us from <br><br> 
<a href="https://sites.google.com/site/hpardoe/hacl">
<b>High resolution automated labelling of the hippocampus and amygdala using <br>
a 3D convolutional neural network trained on whole brain 700 µm isotropic 7T MP2RAGE MRI
</b>
</a>
<br><br>
<hr>
<b>Actual Image Segmentation for  Pardoe-Hippocampus-Amygdala Images of 512x600 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks.<br><br>
<b>class_color_map={Hippocampus:cyan, Amygdala:yellow} </b>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/images/10001_120.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/masks/10001_120.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_output/10001_120.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/images/10005_125.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/masks/10005_125.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_output/10005_125.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/images/10008_153.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/masks/10008_153.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_output/10008_153.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<h3>1. Dataset Citation</h3>
The dataset used here was derived from <br><br> 
<a href="https://sites.google.com/site/hpardoe/hacl">
<b>High resolution automated labelling of the hippocampus and amygdala using <br>
a 3D convolutional neural network trained on whole brain 700 µm isotropic 7T MP2RAGE MRI
</b>
</a>
<br>
<br>
The following explanation was taken from the above web site.
<br><br>
MRI scans, manual hippocampal and amygdala labels and CNN-based labels (5.6 Gb) can be downloaded from NITRC using the curl software tool:
<pre>
curl https://hacl.projects.nitrc.org/pardoe_cnnmp2rage_labelling_mri_scans_20200926.tar.gz -o 
pardoe_cnnmp2rage_labelling_mri_scans_20200926.tar.gz
</pre>
<br>
Code for training the CNN using DeepMedic and analysis of output data: https://github.com/hpardoe/hacl
<br><br>
<b>Data Files</b><br>
<pre>
Whole brain T1-weighted MP2RAGE, defaced :  sub-*_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz
Manual hippocampus and amygdala labels:     sub-*_acq-MP2R700um7T_hipp_amyg_reorient.nii.gz
Brain mask:                                 sub-*_acq-MP2R700um7T_T1w_defaced_reorient_brain_mask.nii.gz
Whole brain T1-weighted MP2RAGE, normalized:sub-*_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd.nii.gz
CNN output, probilistic:                    sub-*_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_ProbMapClass*.nii.gz
CNN 'hard' hippocampus and amygdala labels: sub-*_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_Segm.nii.gz
</pre>
<br>
<b>Citation</b><br>
If you use this dataset please cite our work:
<pre>
  Pardoe, H.R., Antony, A.R., Hetherington, H., Bagić, A.I., Shepherd, T.M., Friedman, D., Devinsky, O. and Pan J. <br>
  (2021), High resolution automated labeling of the hippocampus and amygdala using a 
  3D convolutional neural network trained 
  on whole brain 700 µm isotropic 7T MP2RAGE MRI, Human Brain Mapping, <br>
  doi: https://doi.org/10.1002/hbm.25348
</pre>
<br>
<b>License</b><br>
Unknown
<br>
<h3>
<a id="2">
2 Pardoe-Hippocampus-Amygdala ImageMask Dataset
</a>
</h3>
<h4>2.1 Download ImageMask Dataset</h4>
 If you would like to train this Pardoe-Hippocampus-Amygdala Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1jR8leWcaQ5nJ_mFvCxgUWBRCU79alWXo/view?usp=sharing">
Pardoe-Hippocampus-Amygdala-ImageMask-Dataset.zip
</a> with colorized masks,  
expand the downloaded dataset, and put it under <b>./dataset</b> folder to be:
<pre>
./dataset
└─Pardoe-Hippocampus-Amygdala
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Pardoe-Hippocampus-Amygdala Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/Pardoe-Hippocampus-Amygdala_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<h4>2.2 Derivation of ImageMask Subset</h4>
The folder structure of <b>pardoe_cnnmp2rage_labelling_mri_scans_20200926</b> is the following.<br>
<pre>
./pardoe_cnnmp2rage_labelling_mri_scans_20200926
  ├─sub-control01
  │  ├─anat
  │  │    ├─sub-control01_acq-MP2R700um7T_hipp_amyg_reorient.nii.gz
  │  │    ├─sub-control01_acq-MP2R700um7T_T1w_defaced_reorient_brain_mask.nii.gz
  │  │    ├─sub-control01_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_ProbMapClass1.nii.gz
  │  │    ├─sub-control01_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_ProbMapClass2.nii.gz
  │  │    ├─sub-control01_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_ProbMapClass3.nii.gz
  │  │    ├─sub-control01_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_ProbMapClass4.nii.gz
  │  │    ├─sub-control01_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_Segm.nii.gz
  │  │    ├─sub-control01_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd.nii.gz
  │  │    ├─sub-control01_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz
...
  ├─sub-control02
  │  ├─anat
...
  └─sub-epilepsy17
      └─anat
           ├─sub-epilepsy17_acq-MP2R700um7T_hipp_amyg_reorient.nii.gz
           ├─sub-epilepsy17_acq-MP2R700um7T_T1w_defaced_reorient_brain_mask.nii.gz
           ├─sub-epilepsy17_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_ProbMapClass1.nii.gz
           ├─sub-epilepsy17_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_ProbMapClass2.nii.gz
           ├─sub-epilepsy17_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_ProbMapClass3.nii.gz
           ├─sub-epilepsy17_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_ProbMapClass4.nii.gz
           ├─sub-epilepsy17_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd_pred_Segm.nii.gz
           ├─sub-epilepsy17_acq-MP2R700um7T_T1w_defaced_reorient_subtrMeanDivStd.nii.gz
           └─sub-epilepsy17_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz        

</pre>
We used a simple Python script to generate our 512x600 pixels upscaled PNG dataset
with colorized masks from  all pairs of <b>*_T1w_defaced_reorient.nii.gz</b> 
and corresponding <b>*_T1w_defaced_reorient_subtrMeanDivStd_pred_Segm.nii.gz</b> in <b>anat</b> sub directories of 
<b>sub_*</b> directories. You may use manual annotation files <b>*_hipp_amyg_reorient.nii.gz</b> instead of predicted segmentation files 
 <b>*_T1w_defaced_reorient_subtrMeanDivStd_pred_Segm.nii.gz</b>. 
<br><br>
For simplicity, we excluded all empty black masks and their corresponding images to generate our PNG dataset, which were 
irrelevant to train our segmentation model, 
and upscaled all images and masks to 512x600 pixels from the original 256x300 pixels.<br>
While the original dataset had four annotation classes (Left-Hippocampus, Right-Hippocampus, Left-Amygdala, and Right-Amygdala), 
we reduced the number of classes to two (Hippocampus and Amygdala) to generate our PNG dataset.
<br><br>

<h4>2.3 Image and Mask samples</h4>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained Pardoe-Hippocampus-Amygdala TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to <b>./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala</b> foder, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large <b>num_layers = 8</b> (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 3
base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 5
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Pardoe-Hippocampus-Amygdala 1+4 classes.<br>
<pre>
[mask]
mask_datatyoe    = "categorized"
mask_file_format = ".png"
;Pardoe-Hippocampus-Amygdala rgb color map dict for 1+2 classes.
rgb_map = {(0,0,0):0, (255,255,0):1,(0,255,255):2}

</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>
By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 13,14,15)</b><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 28,29,30)</b><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 30.<br><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/asset/train_console_output_at_epoch30.png" width="1024" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/eval/train_losses.png" width="520" height="auto"><br>

<br>
<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for Pardoe-Hippocampus-Amygdala.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/asset/evaluate_console_output_at_epoch30.png" width="1024" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Pardoe-Hippocampus-Amygdala/test was very low, and dice_coef_multiclass very high as shown below.
<pre>
categorical_crossentropy,0.0043
dice_coef_multiclass,0.9976
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala</b> folder, and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Pardoe-Hippocampus-Amygdala.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Pardoe-Hippocampus-Amygdala Images of 512x600 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks.<br><br>
<b>class_color_map={Hippocampus:cyan, Amygdala:yellow} </b>
<br>
<table>
<tr>
<th>Input: Image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/images/10001_122.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/masks/10001_122.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_output/10001_122.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/images/10002_104.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/masks/10002_104.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_output/10002_104.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/images/10002_129.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/masks/10002_129.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_output/10002_129.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/images/10003_152.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/masks/10003_152.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_output/10003_152.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/images/10005_125.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/masks/10005_125.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_output/10005_125.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/images/10007_150.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test/masks/10007_150.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_output/10007_150.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
6 3D Volume Segmentation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala</b> folder, and run the following bat file to infer images segmentation for 2D slices of 3D volume NIfTI files
 by the Trained-TensorFlowFlexUNet model for Pardoe-Hippocampus-Amygdala.<br>
<pre>
>./5.infer3d.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNet3DInferencer.py ./train_eval_infer.config
</pre>

<b>infer3d section </b> in <a href="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/train_eval_infer.config">
train_eval_infer.config
<a></b>
<pre>
[infer3d] 
;Specify an images_dir which contains NIfTI files
images_dir    = "./mini_test_3d/images/"
output_dir    = "./mini_test_3d_output/"
slice_shape_order = "hwd"
slice_resize   = (512,600)
slice_rotation = cv2.ROTATE_90_CLOCKWISE 
mask_overlay  = True
</pre>
<hr>
<b>Actual Image Segmentation for 2D Slices of a Pardoe-Hippocampus-Amygdala folder NIfTI</b><br>
Some Slices, Inferred Masks and Mask overlays for a 3D volume <b>sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz</b> file.
<br>
<br>
<b>class_color_map={Hippocampus:cyan, Amygdala:yellow} </b>
<br>
<table>
<tr>
<th>Input: Slice</th>
<th>Prediction: Inferred mask</th>
<th>Mask Overlay</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/slices/10102.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/masks/10102.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/overlays/10102.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/slices/10108.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/masks/10108.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/overlays/10108.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/slices/10115.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/masks/10115.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/overlays/10115.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/slices/10120.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/masks/10120.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/overlays/10120.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/slices/10122.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/masks/10122.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/overlays/10122.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/slices/10127.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/masks/10127.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/mini_test_3d_output/sub-control10_acq-MP2R700um7T_T1w_defaced_reorient.nii.gz/overlays/10127.png" width="320" height="auto"></td>
</tr>
</table>
<hr>

<h3>
7 MaskOverlay Video of 3D Volume Segmentation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala</b> folder, and run the following bat file 
to generate <b>overlays.mp4</b> or <b>overlay.gif</b> for MaskOverlays of 3D Volume Segmentation. <br>
<pre>
>./6.video3d.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/MaskOverlayVideoGenerator.py ./train_eval_infer.config
</pre>
<br>

<b>infer3d section </b> in <a href="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/train_eval_infer.config">
train_eval_infer.config
<a></b>

<pre>
[infer3d] 
mask_overlay  = True
;Specify ".mp4" or ".gif".
;video_fileformat  = ".mp4"
video_fileformat  = ".gif"
</pre>
<br>

<img src="./projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/video_3d/overlays.gif">

 <!--
<video  src="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Pardoe-Hippocampus-Amygdala/tree/main/projects/TensorFlowFlexUNet/Pardoe-Hippocampus-Amygdala/video_3d/overlays.mp4" 
 controls="controls" style=
width="600" height="600">
</video>
-->
<br>
<br>
<h3>
References
</h3>
<b>1. High resolution automated labeling of the hippocampus and <br>
 amygdala using a 3D convolutional neural network trained on whole <br>
 brain 700 μm isotropic 7T MP2RAGE MRI</b><br>
Heath R. Pardoe, Arun Raj Antony, Hoby Hetherington, Anto I. Bagić, Timothy M. Shepherd, <br>
Daniel Friedman, Orrin Devinsky, Jullie Pan<br>
<a href="https://onlinelibrary.wiley.com/doi/10.1002/hbm.25348">https://onlinelibrary.wiley.com/doi/10.1002/hbm.25348"</a>
<br><br>
<b>2. TensorFlow-FlexUNet-Image-Segmentation-RISE-MICCAI-LISA-Hippocampus-T2W</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-RISE-MICCAI-LISA-Hippocampus-T2W">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-RISE-MICCAI-LISA-Hippocampus-T2W</a>
<br>
<br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai<br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
