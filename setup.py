from setuptools import setup, find_packages

setup(
    name='imagewiz',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'matplotlib',
        'tqdm',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'imagewiz=imagewiz.filters:main',
        ],
    },
    author='Priyam Dalwadi, Nikhil Ajay Kadalge',
    maintainer='Priyam Dalwadi, Nikhil Ajay Kadalge',
    description='A one-line image filter visualization tool for beginners',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    keywords=['image processing', 'filters', 'cv2', 'visualization'],
)
