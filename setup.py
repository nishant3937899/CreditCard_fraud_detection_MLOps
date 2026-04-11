import setuptools

with open('README.md','r',encoding='utf-8') as f:
    long_description = f.read()

version='0.0.0'

REPO_Name='Creditcard_Fraund_detection_MLOps'
AUTHOR_USER_NAME='Nishant Chandra Verma'
SRC_REPO = "MLOps_project"
AUTHOR_EMAIL='nishantchandra987@gmail.com'


setuptools.setup(
    name=SRC_REPO,
    version=version,
    author= AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='This is an MLOps project that is based on the creditcard dataset from kaggle',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nishant3937899/CreditCard_fraud_detection_MLOps',
    package_dir={"":'src'},
    packages=setuptools.find_packages(where="src")
)