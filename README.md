# python-logisticregression

Logistic regression is a simple 
![test](https://latex.codecogs.com/gif.latex?x=x^3)
# Description

# Resources
## Basic Algorithm 

[cost function](http://latex.codecogs.com/gif.download?J%28%5Ctheta%29%20%3D%20%5Csum%5E%7Bm%7D_%7Bi%3D1%7D%20Cost%20%28h%28x_i%2C%20%5Ctheta%29%2C%20y_i%29)
[sigmod function](http://latex.codecogs.com/gif.download?%5Csigma%20%28t%29%20%3D%20%5Cfrac%7B1%7D%7B1+e%5E-t%7D)

# Why not use SciPy?
[SciPy](https://scipy.org/) has a k-means [implementation](https://docs.scipy.org/doc/scipy/reference/cluster.vq.html). The objective of this work is to build a pure python implementation for the purposes of learning, and helping others learn the k-means algorithm. Interested readers with only minimal python experience will be able to read, and step over this code without the added complexity of a library such as SciPy. It is not by any means intended for production use :)

# Running the code
## Dependencies
+ python 3.6.3
+ matplotlib 2.1.1 - see [here](https://matplotlib.org/users/installing.html) for installation instructions.

## Execution
Run the code with the python interpreter: 

```python kmeans.py ./resources/<config.cfg>```

Where config.cfg is a plain text configuration file. The format of the config file is a python dict with the following fields:

```
{

   'data_file' : '\\resources\\2d_1.csv',
   'data_project_columns' : ['x1', 'x2'],
   'class_label_col' : ['y'],
   'class_label_mapping' : {1 : 1, 0 : 0},
   'data_prep_func' : 'unit_normalize',
   'learning_rate' : 0.025,
   'plot_func' : 'plot_simple_two_dimensional',
   'plot_config' : {'colors' : {1 : 'green', 0: 'red'},
                    'x-axis-att' : 'x1',
                    'y-axis-att' : 'x2',
                    'output_file_prefix' : '2d_1' }
 }
```

You have to specify:
 + a csv data file;
 + which columns of data to project;
 + which column specifies the class label;
 + how to map the class label to a the binary label {0,1};
 + learning rate, the adaptation speed of the batch gradient descent;
 + function that can plot data once logistic regression has been completed;
 + plot config depends on the plot func specified
 
# Results
## Basic 2D 1 and Basic 2D 2

