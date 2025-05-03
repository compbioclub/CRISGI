from setuptools import setup, find_packages

setup(
    name='CRISGI',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/compbioclub/CRISGI',
    include_package_data=True,
     package_data={
        'crisgi': ['*.pk'],
    },
    install_requires=[],
    description="...",
    author="",
    author_email="",
    license="MIT"
)
