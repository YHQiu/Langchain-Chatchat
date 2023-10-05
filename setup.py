import os

from setuptools import setup, find_packages
def gen_data_files(*dirs):
    results = []

    for src_dir in dirs:
        for root,dirs,files in os.walk(src_dir):
            results.append((root, map(lambda f:root + "/" + f, files)))
    return results
setup(
    name='fmchain',
    version='0.0.1',  # 修改为你的项目版本号
    author='hy Qiu',
    author_email='your.email@example.com',
    description='自定义Langchain-Chatchat',
    packages=find_packages(),  # 自动查找和包含所有的Python包
    include_package_data=True,
    data_files = gen_data_files("docs", "lib", "js", "jpg", "png", "jpeg"),
    install_requires=[
        'langchain==0.0.287',
        'fschat[model_worker]==0.2.28',
        'openai',
        'sentence_transformers',
        'transformers>=4.31.0',
        'torch~=2.0.0',
        'fastapi~=0.99.1',
        'nltk~=3.8.1',
        'uvicorn~=0.23.1',
        'starlette~=0.27.0',
        'pydantic~=1.10.11',
        'unstructured[all-docs]>=0.10.4',
        'python-magic-bin; sys_platform == "win32"',
        'SQLAlchemy==2.0.19',
        'faiss-cpu',
        'accelerate',
        'spacy',
        'PyMuPDF==1.22.5',
        'rapidocr_onnxruntime>=1.3.2',
        'requests',
        'pathlib',
        'pytest',
        'scikit-learn',
        'numexpr',
        # Uncomment the following lines if you want to use corresponding vector store
        # 'pymilvus==2.1.3',
        # 'psycopg2',
        # 'pgvector',
        'numpy~=1.24.4',
        'pandas~=2.0.3',
        'streamlit>=1.26.0',
        'streamlit-option-menu>=0.3.6',
        'streamlit-antd-components>=0.1.11',
        'streamlit-chatbox>=1.1.6,<=1.1.7',
        'streamlit-aggrid>=0.3.4.post3',
        'httpx~=0.24.1',
        'watchdog',
        'tqdm',
        'websockets',
        # DB
        'mysqlclient',
        'duckduckgo-search',
    ],
)