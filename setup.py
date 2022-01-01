import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simager",
    version="0.1.0",
    author="ulwan",
    license="MIT",
    author_email="ulwan.nashihun@tiket.com",
    description="A small package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pypi.org/project/simager",
    packages=setuptools.find_packages(),
    package_data={"simager": ["data/normalizer.p", "data/stop_w.p"]},
    keywords=["nlp", "text-processing", "machine-learning", "data-scientist", "text-cleaner"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
    ],
    install_requires=[
        "emoji>=0.6.0",
        "beautifulsoup4>=4.9.3",
        "scikit-learn>=1.0.0",
        "imbalanced-learn>=0.8.1",
        "lightgbm>=3.3.1",
        "xgboost>=1.5.0",
        "catboost>=1.0.3",
        "matplotlib>=3.5.1",
        "pandas>=1.3.5",
        "scikit-optimize>=0.9.0",
        "scipy>=1.7.3"
    ],
    python_requires=">=3.6",
)
