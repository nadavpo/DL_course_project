forked from https://github.com/Tensor-king/SA_Uet-pytorch

We work on this project for the Intoduction to Deep Learning course as part of our MSc in Tel-Aviv University.

The main idea is to take small data set, generate relative big new data set from the original data and integrate it into the train process.
We based our code on the above but extreamlly improve it.

## Results

| Dataset  |   SE   |   SP   |  ACC   |  AUC   | F1     |
|----------|:------:|:------:|:------:|:------:|--------|
| base     | 0.8352 | 0.9885 | 0.9774 | 0.9917 | 0.9138 |
| ours     | 0.729  | 0.991  | 0.986  | 0.983  | 0.936  |


## Train and eval

on train.py\inference.py config the parameters on bottom to control the process (can be done alse with command line)

