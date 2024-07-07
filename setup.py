from setuptools import setup, find_packages

setup(
    name='pyperch',
    url='https://github.com/jlm429/pyperch',
    version='0.1.2',
    platforms=['Any'],
    license='New BSD',
    author='John Mansfield',
    author_email='jlm429@gmail.com',
    description='Randomized opt networks with PyTorch',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=[
        'skorch==0.15.0',
        'torch>=2.2, <2.4',
        'numpy',
        'matplotlib'
    ],
)