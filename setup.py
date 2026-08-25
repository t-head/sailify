from setuptools import setup, find_packages


setup(
    name='sailify',
    version='1.0.0',
    description='Generic CUDA to PPU source code conversion tool',
    long_description=open('README.md').read() if __import__('os').path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    author='sailify contributors',
    python_requires='>=3.8',
    packages=find_packages(),
    py_modules=['sailify_cli'],
    package_data={
        'sailify': ['*.tpl', '*.h'],
        'sailify.unsupported_stubs': ['*.json'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'sailify=sailify_cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Code Generators',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
