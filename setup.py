from setuptools import setup, find_packages

# Get the long description from the README file
def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

setup(
    name='hyperct',
    version='0.3.4',
    description='Low memory hypercube triangulations and sub-triangulations',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/stefan-endres/hyperct',
    project_urls={
        'Bug Reports': 'https://github.com/stefan-endres/hyperct/issues',
        'Source': 'https://github.com/stefan-endres/hyperct',
    },
    author='Stefan Endres, Carl Sandrock',
    author_email='stefan.c.endres@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.16.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'pytest-benchmark>=3.4',
        ],
        'plotting': [
            'matplotlib>=3.0',
        ],
        'gpu': [
            'torch>=2.0',
        ],
    },
    keywords=['optimization', 'triangulation', 'simplicial-complex',
              'hypercube', 'computational-geometry'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    zip_safe=False,
)
