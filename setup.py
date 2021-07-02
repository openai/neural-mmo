from pdb import set_trace as T

import os
import shutil
import zipfile
from pathlib import Path

from setuptools import find_packages, setup
import glob

import versioneer

README = Path("README.md").read_text()
REPO_URL = "https://github.com/jsuarez5341/neural-mmo"
VERSION = versioneer.get_version()

current_dir = Path(__file__).resolve().parent


def read_requirements_file(requirements_version):
    with open(
        str(current_dir / "requirements" / f"{requirements_version}.txt")
    ) as reqs_file:
        reqs = reqs_file.read().split()
    lines_to_remove = []
    for idx in range(len(reqs)):
        if "-r " in reqs[idx]:
            lines_to_remove.append(idx)
    for idx in lines_to_remove:
        reqs.pop(idx)
    return reqs

setup(
    name="neural-mmo",
    description="Neural MMO is a massively multiagent environment for artificial intelligence research inspired by "
    "Massively Multiplayer Online (MMO) role-playing games",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    version=VERSION,
    cmdclass=versioneer.get_cmdclass(),
    install_requires=read_requirements_file("base"),
    extras_require={
        "rllib": read_requirements_file("rllib"),
    },
    entry_points={
        "console_scripts": [
            "neural-mmo-forge=neural_mmo.Forge:main",
        ]
    },
    python_requires=">=3.6",
    license="MIT",
    author="Joseph Suarez",
    author_email="sealsuarez@gmail.com",
    url=REPO_URL,
    keywords=["Neural MMO", "MMO"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
)

