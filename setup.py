import setuptools

setuptools.setup(
    name='truckms',
    description='Python package for truck monitoring system',
    version='0.0.1',
    url='https://github.com/Iftimie/TruckMonitoringSystem',
    author='Alexandru Iftimie',
    author_email='iftimie.alexandru.florentin@gmail.com',
    packages=setuptools.find_packages(),
    dependency_links=['https://download.pytorch.org/whl/torch_stable.html',
                      'https://github.com/philferriere/cocoapi/tarball/master#subdirectory=PythonAPI'],
    #https://stackoverflow.com/questions/32688688/how-to-write-setup-py-to-include-a-git-repo-as-a-dependency
    install_requires=[
        'torch==1.3.0+cpu',
        'torchvision==0.4.1+cpu',
        'opencv-contrib-python',
        'opencv-python',
        'matplotlib',
        'pillow',
        'deprecated',
        'flask',
        'sklearn',
        'numpy',
        'pandas',
        'typing',
        'youtube_dl',
        'Flask-Bootstrap4',
        'pytest',
        'progressbar',
        'tqdm',
        'pycocotools',
        'pytest-mock',
        'mock',
        'gputil',
        'sqlalchemy',
        'netifaces'
      ],

    keywords=['object', 'detection', 'truck']
    )
