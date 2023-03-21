from setuptools import setup
setup(
    name='dynamicworld',
    version='1.0.0',    
    description='A simple Python wrapper for the use of DynamicWorld.',
    author='Federico Ricciuti',
    author_email='ricciuti.federico@gmail.com',
    packages=['dynamicworld'],
    install_requires=[
        'tensorflow',
        'numpy',
        'tqdm'               
    ]
)