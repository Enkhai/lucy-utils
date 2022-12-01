from pathlib import Path

import setuptools

with open(str(Path(__file__).parent) + '/requirements.txt', "r") as f:
    install_requires = f.read().splitlines()
    install_requires = [i for i in install_requires if not i.startswith('#')]  # discard comments

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lucy-utils',
    version="0.2",
    author="Will Moschopoulos",
    description="A set of utilities for the Rocket League bot Lucy, useful for"
                " building effective Reinforcement Learning agents for Rocket League, using RLGym.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Enkhai/lucy-utils",
    project_urls={
        "Bug Tracker": "https://github.com/Enkhai/lucy-utils/issues",
    },
    license="MIT",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=[
        "https://download.pytorch.org/whl/cu113"  # for PyTorch install
        "https://data.pyg.org/whl/torch-1.11.0+cu113.html"  # for PyG install
    ]
)
