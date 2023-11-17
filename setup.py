import re
from setuptools import setup, find_packages


def get_version():
    filename = "human_pose_estimator/__init__.py"
    with open(filename) as f:
        match = re.search(
            r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0]
    return version


def get_install_requires():
    with open("requirements.txt") as req:
        return req.read().split("\n")


def get_long_description():
    with open("README.md") as f:
        return f.read()


def main():
    setup(
        name='human_pose_estimator',
        version=get_version(),
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        packages=find_packages(),
        url='https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch',
        license='Apache',
        author='Daniil-Osokin',
        author_email='daniil.osokin@gmail.com',
        description='Lightweight human pose estimation using pytorch',
        install_requires=get_install_requires(),
        package_data={"human_pose_estimator": []},
        entry_points={"console_scripts": ["poseestimator=human_pose_estimator.__main__:main"]},
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Natural Language :: English",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3 :: Only",
        ],
    )


if __name__ == "__main__":
    main()
