from os import path
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

this_directory = path.abspath(path.dirname(__file__))    
# read the contents of requirements.txt
with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setuptools.setup(name='mlgorithms',
      version='0.0.6',
      author='Dong',
      author_email='dongyuncheng1991@gmail.com',
      description='Machine Learning Lib.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/doycode/mlgorithms.git',
      packages=setuptools.find_packages(),
      install_requires=requirements,
      classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: Apache Software License",
              "Operating System :: OS Independent",
              ]
#      license='MIT',
#      zip_safe=False
)