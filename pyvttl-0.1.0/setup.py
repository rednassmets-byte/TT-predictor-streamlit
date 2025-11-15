from setuptools import setup, find_packages

setup(
    name="pyvttl",
    version="0.1.0",
    description="Python wrapper for the VTTL SOAP API",
    author="Tim Jacobs",
    author_email="jacobs.tim@gmail.com",
    url="https://github.com/jacobstim/pyvttl",
    packages=find_packages(),
    install_requires=["zeep"],
    python_requires=">=3.7",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
