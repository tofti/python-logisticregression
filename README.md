# python-logisticregression


# Description

# Resources
## Basic Algorithm 



![\sigma (t) = \frac{1}{1+e^-t}](http://latex.codecogs.com/gif.latex?\sigma&space;(t)&space;=&space;\frac{1}{1&plus;e^-t})

![h(x_i,\theta) = \frac{1}{1+e^{-x_iT\theta}}](http://latex.codecogs.com/gif.latex?h(x_i,\theta)&space;=&space;\frac{1}{1&plus;e^{-x_iT\theta}})


![J(\theta)=\sum^{m}_{i=1} Cost (h(x_i, \theta),y_i)](http://latex.codecogs.com/gif.latex?J(\theta)&space;=&space;\sum^{m}_{i=1}&space;Cost&space;(h(x_i,&space;\theta),&space;y_i))


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

