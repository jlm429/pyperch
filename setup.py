from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pyperch",
    version="0.2.2",
    url="https://github.com/jlm429/pyperch",
    author="John Mansfield",
    author_email="jlm429@gmail.com",
    description="Randomized optimization networks with PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    platforms=["Any"],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    
    # --- Python version support ---
    python_requires=">=3.10",

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "scikit-learn>=1.7.2,<2.0.0",
        "optuna>=3.6.0,<4.0.0",
    ],
)
