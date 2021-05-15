from setuptools import setup, find_packages

# metadata
NAME = "AppliedTransformers"
version = "0.0.1"
author = "Niclas Doll"
author_email = "niclas.doll@amazonis.net"

description = "SOTA Transformer Models for NLP tasks"
# readme as long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = "\n" + f.read()

setup(

    name =          "AppliedTransformers",
    version =       "0.0.1",
    author =        "Niclas Doll",
    author_email =  "niclas.doll@amazonis.net",

    description = description,
    long_description = long_description,
    long_description_content_type = "text/markdown",

    url = "https://github.com/ndoll1998/AppliedTransformers",
    project_urls = {
        "Bug Tracker": "https://github.com/ndoll1998/AppliedTransformers/issues"
    },

    install_requires = [
        "requests>=2.25.1",
        "transformers>=4.2.2",
        "scikit-learn>=0.24.1",
        "matplotlib>=3.3.3",
        "tqdm>=4.56.0",
        "pandas>=1.2.1"
    ],
    packages = find_packages(exclude=["test", "examples"]),

    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Indipendent"
    ]    

)
