from setuptools import setup, find_packages


# with open('README.rst') as f:
#     readme = f.read()
#
# with open('LICENSE') as f:
#     license = f.read()

setup(
    name='eritlux',
    version='0.1.0',
    description='parameter evolution fitting',
    # long_description=readme,
    author='Jussi Kuusisto & Stephen Wilkins',
    author_email='s.wilkins@sussex.ac.uk',
    url='https://github.com/stephenmwilkins/eritlux',
    # license=license,
    packages=find_packages(exclude=('examples','docs'))
)
