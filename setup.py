from setuptools import setup, find_packages

setup(
    name="rl_arena",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "gymnasium",
        "ale_py",
        "numpy",
        "opencv-python",
    ],
)
