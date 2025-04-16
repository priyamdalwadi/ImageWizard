from setuptools import setup, find_packages

setup(
    name='imagewiz',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'matplotlib'
    ],
    author='Priyam Dalwad',
    description='A one-line image filter visualization tool for beginners',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    keywords=['image processing', 'filters', 'cv2', 'visualization'],
)
