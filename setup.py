from setuptools import find_packages, setup
from nmmo import version

REPO_URL = "https://github.com/neuralmmo/environment"

setup(
    name="nmmo",
    description="Neural MMO is a platform for multiagent intelligence research inspired by "
    "Massively Multiplayer Online (MMO) role-playing games. Documentation hosted at neuralmmo.github.io.",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    version=version,
    install_requires=[
        'pytest-benchmark',
        'fire==0.4.0',
        'setproctitle==1.1.10',
        'autobahn==19.3.3',
        'Twisted==19.2.0',
        'gym==0.17.2',
        'vec-noise==1.1.4',
        'imageio==2.8.0',
        'tqdm==4.61.1',
    ],
    extras_require={
        'docs': [
            'sphinx-rtd-theme==0.5.1',
        ],
        'rllib': [
            'ray==1.5.2',
            'tensorflow==2.4.1',
            'dm-tree==0.1.5',
            'torch'
        ],
    },
    python_requires=">=3.8",
    license="MIT",
    author="Joseph Suarez",
    author_email="jsuarez@mit.edu",
    url=REPO_URL,
    keywords=["Neural MMO", "MMO"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    include_package_data=True,
)

