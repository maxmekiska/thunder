from setuptools import setup, find_packages


setup(
    author="Maximilian Mekiska",
    name="thunder",
    version="0.0.1",
    packages=find_packages(include=["thunder", "thunder.*"]),
    install_requires=[
        "uvicorn==0.32.1",
        "fastapi==0.115.5",
        "pandas==2.2.3",
        "torch==2.5.1",
        "scikit-learn==1.5.2",
        "pytorch-lightning==2.4.0",
        "click==8.1.7",
    ],
    python_rquieres=">= 3.10.0, <= 3.11.0",
)
