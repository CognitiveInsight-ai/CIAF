"""
Setup script for the Cognitive Insight AI Framework (CIAF) package.
"""

from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name='ciaf',  # The new package name
    version='0.1.0',  # Start with a small version number
    author='CIAF Development Team',
    author_email='ciaf@example.com',
    description='Cognitive Insight AI Framework (CIAF) - A modular framework for verifiable AI with provenance tracking.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your_username/ciaf',  # Replace with your GitHub repo
    packages=find_packages(),  # Automatically find all packages in the directory
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Or your chosen license
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',  # Or 4 - Beta, 5 - Production/Stable
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Security :: Cryptography',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',  # Minimum Python version required
    install_requires=install_requires,  # Dependencies from requirements.txt
    keywords='ciaf cognitive insight ai artificial-intelligence transparency provenance cryptography framework',
)
