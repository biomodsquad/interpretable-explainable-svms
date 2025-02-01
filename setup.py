from setuptools import setup, find_packages

setup(
    name='mistic',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
    ],
    author='BioModSquad',
    author_email='group@biomodsquad.org',
    description='Interpretable and explainable support vector machines',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/biomodsquad/interpretable-explainable-svms',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)