from setuptools import setup, find_packages

setup(
    name='LangRAGeval',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'langchain',
        'pydantic',
    ],
    author='Prashant Verma',
    author_email='prashant27050@gmail.com.com',
    description='A library for Langchain based evaluating RAG.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/RAGeval',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
