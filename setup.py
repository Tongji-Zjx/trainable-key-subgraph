from setuptools import find_packages, setup


setup(
    name="keysubgraph",
    version="0.1.0",
    description="Trainable signed key-subgraph extraction and structural analysis",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.7,<3.8",
)
