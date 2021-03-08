from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='summarizers',
    version='1.0.0',
    description='opensource text summarization toolkit.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Hyunwoong Ko',
    author_email='gusdnd852@naver.com',
    url='https://github.com/summarizers/summarizers',
    install_requires=[
        'transformers>=4',
        'torch',
    ],
    packages=find_packages(exclude=['gym']),
    python_requires='>=3',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
    ],
)