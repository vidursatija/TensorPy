import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorpy",
    version="1.0.0",
    author="Vidur Satija",
    author_email="vidursatija@gmail.com",
    description="A simple numpy autodiff library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vidursatija/TensorPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
