import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name='mlgorithms',
      version='0.0.5',
      author='Dong',
      author_email='dongyuncheng1991@gmail.com',
      description='Machine Learning Lib.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/doycode/mlgorithms.git',
      packages=setuptools.find_packages(),
      classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: Apache Software License",
              "Operating System :: OS Independent",
              ]
#      license='MIT',
#      zip_safe=False
)