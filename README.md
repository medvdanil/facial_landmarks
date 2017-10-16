## Required packages
- Python 3.5
- Tensorflow 1.3.0
- NumPy 1.13.3
- DLIB 19.7.0
- matplotlib 2.1.3



## Training

Run `python3 train.py [data_directory]` to generate cropped data and train a model. 
The data will be saved to `data.npz`. Next time, the training will start immediately. Delete `data.npz` to generate it again.
`data_directory` must contain subdirectories (`300W` and `Menpo`) with `train` and `test` folders in them.

## Testing

Run `python3 test.py` to launch a demonstration of the algorithm.


