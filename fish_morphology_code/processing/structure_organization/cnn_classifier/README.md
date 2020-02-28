
model_training.py: the function I called to train the model and generate the accuracy/loss graphs

functions.py: most of the actual code used for data-loading/handling and model generation during development. Most of what you will probably want are in the first two sections titled "Data Loading Tools" and "Dataloader". These were used to create the image patches used for training the model and for feeding to the model. The code for the final model is also here, under the class name `myoCNN_ResNet_18` (line 461). The last section, called "Run model functions" (line 558), is all the code for training/validating the image

myocyte_CNN_classifier.py: This is the more finalized and packaged version of the model that we gave to modeling way back when. It's also a lot better commented.


Should we add the data used for training? It is about 110Mb.