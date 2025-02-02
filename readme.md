


conda create --name agent_summary

#For installing the Langchain module associated with OpenAI LLM model
conda install langchain
conda install conda-forge::langchain-openai
#For installing the Python library responsible for PDF upload
conda install pypdf
#For installing the library responsible for web app development
conda install conda-forge::streamlit
#For installing tokeniser library that asists with converting text strings into tokens recognizable by OpenAI models
conda install conda-forge::tiktoken
#For installing the library used for invoking the environment file containing secret API key
conda install conda-forge::python-dotenv
conda install conda-forge::pip

pip install -qU langchain-openai
pip install -U langchain-community
pip install -U  pypdf



