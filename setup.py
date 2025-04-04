from setuptools import setup, find_packages

setup(
    name="rare_event_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.6.1",
        "numpy>=2.0.0",
        "torch>=2.0.0, <3.0",
        "torchvision>=0.15.0, <1.0",
        "joblib>=1.3.1",
        "pandas>=2.0.3",
        "matplotlib>=3.7.1",
        "fabio",
        "opencv-python>=4.7.0",
        "scipy>=1.11.2",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ]
)