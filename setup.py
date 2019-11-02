from io import open
from setuptools import find_packages, setup

setup(
    name="ZEN",
    version="0.1.0",
    author="chenguimin, Sinovation Ventures AI Institute authors",
    author_email="chenguimin@chuangxin.com",
    description="a BERT-based Chinese (Z) text encoder Enhanced by N-gram representations",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='NLP deep learning transformer pytorch BERT ZEN',
    license='Apache',
    url="https://github.com/sinovation/ZEN",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['numpy',
                      'boto3',
                      'requests',
                      'tqdm',
                      'regex'],
    entry_points={
      'console_scripts': [
      ]
    },
    # python_requires='>=3.5.0',
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
