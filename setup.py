from setuptools import setup, find_packages

setup(
    name="neural-jump-ode",
    version="0.1.0",
    description="Neural Jump ODE for modeling jump-diffusion processes",
    author="Alexander Dybdahl",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.20.0", 
        "matplotlib>=3.3.0",
        "scipy>=1.7.0"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)