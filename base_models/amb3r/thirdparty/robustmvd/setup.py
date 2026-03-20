# References:
#   https://github.com/HengyiWang/amb3r/blob/main/thirdparty/robustmvd/setup.py

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('rmvd/version.py').read())
setup(
    name='rmvd',
    version=__version__,
    description='Robust Multi-view Depth Framework',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/robustmvd/robustmvd',
    author='Philipp Schröppel',
    author_email='schroepp@cs.uni-freiburg.de',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='pytorch models datatasets depth estimation multi-view benchmark',
    packages=find_packages(exclude=['sample_data', 'tests', 'weights']),
    include_package_data=True,
    install_requires=['torch >= 1.9.0', 'numpy', 'pillow', 'matplotlib', 'pandas', 'pytoml', 'tqdm', 'opencv-python', 'scikit-image'],
    python_requires='>=3.8',
)
