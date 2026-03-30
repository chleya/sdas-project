from setuptools import setup, find_packages

setup(
    name="sdas",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26",
        "matplotlib",
        "gymnasium[minigrid]",
        "stable-baselines3",
        "cleanrl"
    ],
    author="SDAS Team",
    author_email="",
    description="Structure-Driven Agent System",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/chleya/sdas-project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)