from pathlib import Path
from setuptools import find_packages, find_namespace_packages, setup

ROOT = Path(__file__).parent


def find_requirements(filename):
    with (ROOT / filename).open() as requirements_file:
        lines = map(str.strip, requirements_file)

        return [line for line in lines if not line.startswith("#")]


setup(
    name="CPFlow",
    version="0.1.0",
    author="CWHuang",
    packages=find_packages("./CPFlow"),
    description="Convex Potential Flow package",
    include_package_data=True,
    install_requires=find_requirements("requirements.txt"),
    entry_points={},
)