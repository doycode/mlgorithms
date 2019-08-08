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
We have also deployed the project to PyPI, and you can install it anytime, anywhere through the following instruction.

```
pip install mlgorithms
```


## Running the tests

If you install successfully, you can test it with the following code.

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


### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
