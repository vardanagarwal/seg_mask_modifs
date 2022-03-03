import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seg_mask_modifs",
    version="1.0.0",
    author="Vardan Agarwal",
    author_email="vardanagarwal16@gmail.com",
    description="A package for easy generation of mask of different labels using multiple models"
                " and applying different operations on that.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vardanagarwal/seg_mask_modifs",
    project_urls={
        "Bug Tracker": "https://github.com/vardanagarwal/seg_mask_modifs/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['seg_mask_modifs'],
    python_requires=">=3.6",
    install_requires=['torch>=1.10.1',
                      'torchvision>=0.11.2',
                      'numpy>=1.21.4',
                      'googledrivedownloader>=0.4',
                      'Pillow>9.0.0'],
)
