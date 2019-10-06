import setuptools

setuptools.setup(
    name='truckms',
    description='Python package for truck monitoring system',
    version='0.0.1',
    url='https://github.com/Iftimie/TruckMonitoringSystem',
    author='Alexandru Iftimie',
    author_email='iftimie.alexandru.florentin@gmail.com',
    packages=setuptools.find_packages(),
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html'],
    install_requires=[
        'torch==1.2.0+cpu',
        'torchvision==0.4.0+cpu',
        'opencv-contrib-python',
        'opencv-python',
        'matplotlib',
        'pillow',
        'deprecated',
        'flask',
        'sklearn'
      ],

    keywords=['object', 'detection', 'truck']
    )
