# Group6-BrainTumor

## Dependencies
- To run the pipeline.py and efficientnet_tuning.ipynb you can refer to the requirement.txt file, where the list of necessary dependencies is provided.

## Dataset
- The dataset we used can be found using this link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- Overall, there are 7023 images, with 4 classes: glioma, meningioma, notumor, pituitary

## Image folders
- "dataset" - these are the original images (training & test)
- "confusion_matrix" - the confusion matrices for each model
- "misclassified" - 10 correctly classified vs miclassified images
- "feature_importance" - the Grad-CAM feature importance for the same 10 images in "misclassified" folder showing regions of importance
- "models_saved" - the saved weights for each model

## Coding files
- "pipeline.py"
    - this contains the main methods to run all models
    - also saves all the respective images in their folders
    - however, if you run pipeline to save the model weights again, this is quite slow,
    about 1hr for each model with CPU.
- "googlenet.ipynb"
    - this file gives the results for GoogleNet, essentially if you want to run it faster using the GPU
- "efficientnet_tuning"
    - this similarly provides additional results like fine-tuning for EfficientNet
    - runs the model faster than the pipeline
 
## KEY NOTE
- If you want to rerun to get all the images, don't delete the folders
- They have to exist to save the images
