"""Installation file."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

setup(
    name="WABgen",
    version="1.0.0",
    packages=find_packages(),
    install_requires=install_requires,
    author="Orlando Villegas",
    author_email="ovillegas.bello0317@gmail.com",
    description='Wycoff alignment block generator.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/ovillegasb/WABgen',
    classifiers=[  # Clasificadores opcionales
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            'wabgen=wabgen.__main__:main',
        ],
    },
    package_data={
        'wabgen': ['data/*'],
    },
    python_requires='>=3.10'
)
