# Artist Identification
Identify whether a work of art was created by Pablo Picasso or Vincent van Gogh. The dataset provided has a total of 2,000 images. Close to half of the images were created by Picasso and the other half were created by van Gogh.

### Link to dataset
http://public.skinio.com/ml-internship-challenge/artist_dataset.zip

## How to run
Download the dataset from the link above. <br />
NOTE: There is a .zip file in the 'Picasso' folder which must be removed before the .py file is run 

To run the code, execute the following line in terminal: `python artist_identification.py` <br />
NOTE: Files followed by '2' are for the model without data augmentation. eg: `model_best2.hdf5` is the model made *without* data augmentation. `model_best.hdf5` is the model made *with* data augmentation.

To train the model, set `train = True` in *main* function on line #*262* (line # might change in future commits).

### Layout of files in directory
|__ Main directory <br />
&nbsp; &nbsp; &nbsp;    |__ artist_dataset <br />
&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;          |__  Picasso <br />
&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;         |__ vanGogh <br />
&nbsp; &nbsp; &nbsp;    |__ artist_identification.py <br />
&nbsp; &nbsp; &nbsp;    |__ X_train.npy <br />
&nbsp; &nbsp; &nbsp;    |__ X_valid.npy <br />
&nbsp; &nbsp; &nbsp;    |__ X_test.npy <br />
&nbsp; &nbsp; &nbsp;    |__ y_train.npy <br />
&nbsp; &nbsp; &nbsp;    |__ y_valid.npy <br />
&nbsp; &nbsp; &nbsp;    |__ y_test.npy <br />
&nbsp; &nbsp; &nbsp;    |__ model_best.hdf5 <br />


### Required packages
1. Keras
2. numpy
3. opencv
4. tqdm

## Performance
Without Data augmentation:
-	Train accuracy = 90.76389 %
-	Validation accuracy = 83.3334 %
-	Test accuracy = 84.75 %

With Data augmentation:
-	Train accuracy = 91.875 %
-	Validation accuracy = 83.3334 %
-	Test accuracy = 82.25 %

