# mlgorithms

Machine learning libraries implemented entirely in python.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

```
Python 3.
```

### Installing

#### Method 1
Clone the project locally and enter the project folder.

```
pip install .
```

#### Method 2
We have also deployed the project to [PyPI](https://pypi.org/project/mlgorithms/), and you can install it anytime, anywhere through the following instruction.

```
pip install mlgorithms
```


## Running the tests

If you install successfully, below is the test code for ID3.

```
from mlgorithms.ID3 import ID3

dat = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
features_name = ['f1', 'f2']
model = ID3()
built_tree = model.create_tree(dat, features_name, max_depth=None, min_samples_split=2)
print(model.predict(built_tree, [1,0], features_name))
```

Then you can save and load the built tree through the following code.

```
model.save_built_tree('built_tree.m', built_tree)
load_tree = model.load_built_tree('built_tree.m')
```

## Built With

* [Python 3.6](https://www.python.org/downloads/)


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/doycode/mlgorithms/tags). 

## Authors

* **Yuncheng Dong**
* **Email**: dongyuncheng1991@gmail.com

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* Machine Learning in Action
