
The convolutional neural network (CNN) in highlight of this report
aims to perform a localization box task of the cancer present in the
MRI images. Read the report for a complete explanation.

To train the network:
- load all your MRI data in archive 2 with the JSON label file.
- Edit read_images_store_XY.py to adjust your desired input resolution and number of batches
- Run read_images_store_XY.py
- Edit train.py to adjust your number of hidden channels, kernel size... and hyperparameters
- Don't forget to select the same input resolution as in read_images_store_XY.py
- Run train.py

To test the network run and edit test.py