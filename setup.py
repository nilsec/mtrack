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
        'mtrack.preprocessing',
        'mtrack.postprocessing',
        'mtrack.evaluation',
        'mtrack.mt_utils'
            ],
    install_requires = [
        'numpy',
        'scipy',
        'h5py',
            ],
)   
