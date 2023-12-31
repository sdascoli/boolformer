from setuptools import setup, find_packages

setup(
    name='boolformer',
    version='0.1.9',
    description="Transformers for symbolic regression of Boolean functions",
    author="Stéphane d'Ascoli",
    author_email="stephane.dascoli@gmail.com",
    packages=find_packages(),
    license="MIT",
    install_requires=[
        "sympy",
        "matplotlib",
        "numpy",
        "pandas",
        "requests",
        "scikit-learn",
        "scipy",
        "seaborn",
        "setproctitle",
        "tqdm",
        "wandb",
        "gdown",
        "torch",
        "boolean.py",
        "graphviz",
        "treelib",
        "pmlb",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
