# Overview
A repo containing various sample code snippets to illustrate the ussage of langchain.
# Usage
Just setup a python langchain environment, store your api keys in a local .env file and run the various main_xx.py files
An example for the .env file is in the repo: .env.example

For the environment:
```Create and activate virtual environment
conda create –-name langchain_starter
conda activate langchain_starter
```
Install langchain (will install compatible Python version as well)
```
pip install langchain langchain-openai python-dotenv
```
Optionally
```
pip install langchain-experimental
```
Additional packages depending on the tools you want, e.g. 
```
pip install wolframalpha
```