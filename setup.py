from setuptools import setup,find_packages

setup(
    name="easy_ml",
    version="0.1",
    packages=find_packages(),
    install_requires = [
        'scikit-learn'
    ]
)