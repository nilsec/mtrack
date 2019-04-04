from setuptools import setup

setup(
    name='mtrack',
    version='0.1',
    description='ILP based tracking of microtubules in em image stacks',
    url='https://github.com/nilsec/mtrack',
    author='Nils Eckstein',
    author_email='nilsec@ini.ethz.ch',
    license='MIT',
    packages=[
        'mtrack',
        'mtrack.graphs',
        'mtrack.cores',
        'mtrack.preprocessing',
        'mtrack.evaluation',
        'mtrack.mt_utils'
            ],
    install_requires = [
        'numpy',
        'scipy>=0.18.0',
        'h5py',
        'pymongo',
        'pylp'
            ],
)   
