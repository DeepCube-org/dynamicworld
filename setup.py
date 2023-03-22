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
        'tqdm',
        's2cloudless @ git+https://github.com/sentinel-hub/sentinel2-cloud-detector.git'
    ],
    package_data={'dynamicworld': [
        'model/model/forward/saved_model.pb', 
        'model/model/forward/variables/*',
        'model/model/backward/saved_model.pb', 
        'model/model/backward/variables/*',
    ]}
)