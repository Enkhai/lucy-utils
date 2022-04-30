from pathlib import Path

import setuptools

with open(str(Path(__file__).parent) + '/requirements.txt', "r") as f:
    install_requires = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lucy-utils',
    version="0.1",
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
    install_requires=install_requires
)
