from setuptools import setup, find_packages

setup(
    name='dubins_path_planning',
    version='0.0.1',
    packages=find_packages(),  # assumes your package lives in dubins_path_tracking/
    install_requires=[],
    include_package_data=True,
    zip_safe=False,
)