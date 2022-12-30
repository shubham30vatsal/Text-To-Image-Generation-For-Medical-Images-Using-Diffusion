# Text-To-Image Generation Using Stable Diffusion For Medical Images



## This repositiory aims to fine-tune latest stable diffusion model on ROCO dataset. The details of the repository have been listed below:

1. stable_diffusion_roco.ipynb notebook contains the code to fine-tune ROCO on stable diffusion. Please download ROCO dataset and create the format of dataset as instructed in the notebook.
2. train_text_to_image.py contains the hugging face wrapper for fine-tuning stable diffusion.

3. The folder multilabel-classification contains the code to extract top N concepts(conceptIDs), create a new sub-dataset using only these top N conceptIDs to be used for training.

4.The folder multilabel-classification also contains the code to train modified DenseNet121 (with an extra hidden units in classification layers) and code to inference on test images(generate conceptID for a test image).


Please contact sv2128@nyu.edu/rm5707@nyu.edu for any issues/bugs.
