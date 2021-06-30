import os
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from setuptools import find_packages, setup
from tqdm import tqdm
import glob

import versioneer

README = Path("README.md").read_text()
REPO_URL = "https://github.com/jsuarez5341/neural-mmo"
CLIENT_REPO_URL = "https://github.com/jsuarez5341/neural-mmo-client"
VERSION = versioneer.get_version()

current_dir = Path(__file__).resolve().parent


class TqdmHook(tqdm):
    def update_to(self, blocks_completed=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks_completed * block_size - self.n)


def download_file(url, save_dir):
    filename = url.split("/")[-1]
    target_filename = os.path.join(save_dir, filename)
    print(f"Downloading {url} -> {target_filename}")
    with TqdmHook(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=filename
    ) as t:
        urlretrieve(url, filename=target_filename, reporthook=t.update_to)
    print("Download complete!")
    return target_filename


def package_files(directory):
    if os.path.relpath(directory) != directory:
        return []

    if directory.startswith("__"):
        return []

    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("", path, filename))
        for directory in directories:
            paths += package_files(directory)
    return paths


def extract_zip(zip_file, target_dir):
    print(f"Extracting {zip_file} -> {target_dir}")
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def setup_neural_mmo_client():
    client_archive = None
    client_version = os.getenv("CLIENT_VERSION", VERSION)
    extracted_client_path = f"neural-mmo-client-{client_version}"
    try:
        client_url = f"{CLIENT_REPO_URL}/archive/refs/heads/v{client_version}.zip"
        client_archive = download_file(client_url, current_dir)
        extract_zip(client_archive, current_dir)
        shutil.copytree(
            extracted_client_path, "neural_mmo/forge/embyr", dirs_exist_ok=True
        )
    except Exception as e:
        raise e
    finally:
        if client_archive is None:
            return
        if os.path.exists(client_archive):
            os.remove(client_archive)
        if os.path.exists(extracted_client_path):
            shutil.rmtree(extracted_client_path)


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


if __name__ == "__main__":
    setup_neural_mmo_client()

extra_dirs = ["resource", "baselines", "forge/embyr"]
extra_files = []
for extra_dir in extra_dirs:
    extra_files += list(glob.iglob(str(current_dir / "neural_mmo" / extra_dir) + "/**", recursive=True))

setup(
    name="neural-mmo",
    description="Neural MMO is a massively multiagent environment for artificial intelligence research inspired by "
    "Massively Multiplayer Online (MMO) role-playing games",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "": extra_files,
    },
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
