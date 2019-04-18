# Artist Identification
Identify paintings of Picasso and vanGogh

### Link to dataset
http://public.skinio.com/ml-internship-challenge/artist_dataset.zip

### How to run
Download the dataset from the link above. <br />
NOTE: There is a .zip file in the 'Picasso' folder which must be removed before the .py file is run

To run the code, execute the following line in terminal: `python artist_identification.py` 

To train the model, set `train = True` on line _262_.

### Layout of files in directory
- Main directory <br />
- - artist_dataset <br />
- - -  Picasso <br />
- - - vanGogh <br />
- - artist_identification.py <br />
- - X_train.npy <br />
- - X_valid.npy <br />
- - X_test.npy <br />
- - y_train.npy <br />
- - y_valid.npy <br />
- - y_test.npy <br />
- - model_best.hdf5 <br />
