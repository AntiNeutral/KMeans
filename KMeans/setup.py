from setuptools import setup

setup(
    name='KMeans',
    url='https://github.com/AntiNeutral/KMeans',
    author='AntiNeutral',
    author_email='hsy35202@gmail.com',
    packages=['kmeans'],
    install_requires=['numpy', 'torch==2.5.1+cu124'],
    version='0.0.1',
    license='MIT',
    description='KMeans with PyTorch'
)
