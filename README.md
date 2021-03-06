# mlgorithms

Machine learning libraries implemented entirely in python. Updating...

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

* [Python 3.6](https://www.python.org/downloads/)

### Installing

#### Method 1
The project was deployed to [PyPI](https://pypi.org/project/mlgorithms/), and it is recommended to use pip for installation:

```
pip install mlgorithms            # normal install
pip install --upgrade mlgorithms  # or update if needed
```

#### Method 2
Alternatively, you could clone and run setup.py file:

```
git clone https://github.com/doycode/mlgorithms.git
cd mlgorithms
pip install .
```



## Running the tests

If you install successfully, below is the test code for ID3.

```
from mlgorithms.ID3 import ID3

X = [[1,1],[1,1],[1,0],[0,1],[0,1]]
y = ['yes', 'yes', 'no', 'no', 'no']
features_name = ['f1', 'f2']
model = ID3(features_name=features_name)
built_tree = model.fit(X, y)
print(model.predict(built_tree, [[1,0]]))
```

Then you can save and load the built tree through the following code:

```
model.save_built_tree('built_tree.m', built_tree)
load_tree = model.load_built_tree('built_tree.m')
```


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/doycode/mlgorithms/tags). 

## Authors

* **Yuncheng Dong**
* **Email**: dongyuncheng1991@gmail.com

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](https://github.com/doycode/mlgorithms/blob/master/LICENSE) file for details.

## Acknowledgments

* Peter Harrington
