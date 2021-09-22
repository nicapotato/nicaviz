import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nicaviz",
    version="0.0.5",
    author="nicapotato",
    author_email="nick.brooks27@gmail.com",
    description="Python Visualization Package",
    long_description="Build over seaborn and matplotlib",
    long_description_content_type="text/markdown",
    url="https://github.com/nicapotato/nicaviz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.14.2",
        "pandas>=0.23.4",
        "python-dateutil>=2.8.1",
        "seaborn>=0.9.0",
        "statsmodels>=0.9.0",
        "wordcloud>=1.8.1 "
    ],
    python_requires='>=3.5',
)
