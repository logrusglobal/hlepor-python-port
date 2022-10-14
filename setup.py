import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hLepor",       # Replace with your own username
    version="0.0.4",
    author="Irina Sorokina, Gleb Erofeev, Serge Gladkoff",
    license="Apache Software License",
    author_email="rd@logrusglobal.com",
    description="This is Python port of original algorithm by Aaron Li-Feng Han",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/logrusglobal/hLepor-python-port",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)
