from setuptools import setup, find_packages

setup(
    name='model_utils',
    packages=find_packages(),
    install_requires=[
        'evaluate',
        'numpy',
        'pandas'
    ],
)