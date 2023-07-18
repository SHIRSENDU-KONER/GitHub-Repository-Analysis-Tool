# Importing libraries
import os
import re
import json
import radon
import pandas as pd
import streamlit as st
import radon.raw
import nbformat
from nbconvert import PythonExporter
from langchain.chat_models import ChatOpenAI
from radon.complexity import cc_visit
from radon.metrics import h_visit
from github import Github
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain import FAISS, PromptTemplate, OpenAI, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DotDict:
    """Converting the dictionary for accessing the contents using dot operator"""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __repr__(self):
        attributes = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attributes})"


def fetch_ipynb_content(content):
    """returnn the ipython code of a .ipynb file"""
    try:
        notebook = nbformat.reads(content, as_version=4)
    except:
        return None

    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(notebook)
    return python_code


def get_text_chunks(text):
    """Takes the text as input and returns chunks of documnets.
    RecursiveCharacterTextSplitter splits the text recursively."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        # language=Language,
        add_start_index=True
    )
    return text_splitter.split_text(text)


def analysze_data(data):
    """return the metrics associated with the code using radon library"""
    try:
        analyzed_data = radon.raw.analyze(data)
        complexity = radon.complexity.ComplexityVisitor.from_code(data).total_complexity
        hal_report = radon.metrics.h_visit(data).total
    except SyntaxError:
        analyzed_data = calculate_metrics(data)
        analyzed_data = DotDict(analyzed_data)
        complexity = 'None'
        hal_report = 'None'

    return analyzed_data, complexity, hal_report


def extract_modules(data, file_format):
    """return the names of the modules and packages used in the file"""
    modules_each_file = set()
    pattern_set = {".py": r'^\s*(?:import\s+(\w+|\w+\.\w+)\b|from\s+(\w+|\w+\.\w+)\s+import\s+([^\n]+))',
                   ".ipynb": r'^\s*(?:import\s+(\w+|\w+\.\w+)\b|from\s+(\w+|\w+\.\w+)\s+import\s+([^\n]+))',
                   ".cpp": r'#include\s+["<]([\w\.]+)[">]',
                   ".c": r'#include\s+["<]([\w\.]+)[">]',
                   ".java": r'import\s+([\w.]+);',
                   ".js": r'(?:import\s+[^;]*?\s+from\s+)?[\'"]([^"\']+)["\']'}
    pattern = pattern_set[file_format]
    import_statements = re.findall(pattern, data, flags=re.MULTILINE)
    if file_format == '.py' or file_format == '.ipynb':
        for statement in import_statements:
            if statement[0]:
                modules_each_file.add(statement[0])
            else:
                modules_each_file.add(statement[1])
                modules_each_file.add(statement[2])
    else:
        modules_each_file = import_statements
    return modules_each_file


def display_repo_names_and_url(username, access_token):
    """return the names of the repos associated with the given Username"""

    # Github Object
    g = Github(access_token)
    user = g.get_user(username)
    cnt = 0
    repo_names = []
    repo_descriptions = {}
    repo_url = []
    for repo in user.get_repos():
        cnt += 1
        repo_names.append(repo.name)
        description = repo.description
        repo_url.append(repo.html_url)
        repo_descriptions[repo.name] = description if description else None

    return user, repo_names, repo_descriptions, repo_url


def calculate_metrics(code):
    """return the metrics associated with the code"""
    lines = code.strip().split('\n')
    loc = len(lines)
    blank = sum(1 for line in lines if not line.strip())
    comments = sum(1 for line in lines if line.strip().startswith('//'))
    multi = sum(1 for line in lines if re.match(r'\s*/\*.*\*/\s*', line))
    single_comments = comments - multi
    sloc = loc - blank - comments - multi
    lloc = sloc - sum(1 for line in lines if line.strip().endswith(';'))

    metrics = {
        'loc': loc,
        'lloc': lloc,
        'sloc': sloc,
        'comments': comments,
        'blank': blank,
        'multi': multi,
        'single_comments': single_comments,
    }

    return metrics


def get_results(openai_api_key):
    """returns the results obtained from using the OpenAI Api key"""
    # Loading the csv file
    loader = CSVLoader(file_path='repo_details.csv', encoding="utf-8")
    data = loader.load()
    data = '\n'.join([doc.page_content for doc in data])

    # Breaking the file into chunks of documents
    docs = get_text_chunks(data)

    # Embedding the documents
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Saving the embedded files into a vector database
    database = FAISS.from_texts(docs, embeddings)

    # Formatting the template for passing it to OpenAI
    template = """You are Super smart Github Repository AI system.You are given a csv file data that have been embedded
        into FAISS.You are asked to find the most technically complex and challenging repository from the given csv.
        Retrieve the data from there and return the repository name that you consider to be the most complex.
        To measure the technical complexity of a GitHub repository, You will analyze and calculate various factors like 
        Cyclomatic Complexity, Halstead Complexity, Maintainability Index and any other factor asked in the question 
        from the source data. Additionally, you will consider the programming languages used, the size of the codebase. 

        {context}

        Answer the question below:

        Question: {question}
        Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    prompt.format(
        context='''You can consider other factors as well if you think they are relevant for determining the technical 
                complexity of a GitHub repository.
                Calculate the complexity score for each repo by assigning weights to each factor and summing up the weighted scores.
                The repo with the highest complexity score will be considered the most technically complex.''',
        question="",
    )
    chain_type_kwargs = {"prompt": prompt}
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
        chain_type="stuff",
        retriever=database.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )
    query = '''Which is the most complex repository depending upon the following factors:
            - No. of Files in the repository
            - Programming languages used in the repository files
            - Complexity of the statements used as code
            - Packages and Modules imported inside the repo (to be calculated from the code)
            - No.of logical lines of code or lines of executable source code
            - Cyclomatic Complexity of the files/codes per repository
            - Halstead Complexity of the files/codes as per repository
            - Maintainability Index of the files/codes as per repository
            Return only the name of the repository, its complexity score (within a scale of 1-100) and the analysis of 
            the repository showing why it is the most technically challenging/Complex repository. Try to provide a 
            detailed analysis to hold your answer strong within 200 words in a paragraph. The output should be in the 
            following format:
            [start a new line]
            Repository Name: <name of the repository>
            [start a new line]
            Repository Link: <link to the repository>
            [start a new line]
            Complexity Score: <complexity score of the repository>
            [start a new line]
            Analysis: <analysis of the repository in a paragraph of about 150 words>'''
    result = chain.run(query)
    # print("The Prompt of the Langchain is :", prompt)

    if not result:
        return "No result."
    else:
        return "\n".join(result.split("."))


def get_each_repo_data(user, repo_name):
    """returns the data from each repo"""
    repo_detail = user.get_repo(repo_name)
    cnt_ = 0
    # fetch the contents of the repo
    try:
        contents = repo_detail.get_contents('')

    except:
        return None, None

    repo_raw_text = {}

    # LIST OF FILES EXTENSIONS THAT ARE TO BE CONSIDERED
    file_extensions_list = ["py", "ipynb", "cpp", "c", "java", "php", 'js']

    while contents:
        file_content = contents.pop(0)
        file_extension = file_content.name.split(".")[-1]

        # IGNORE ".gitignore", ".github", ".folders" lookalike folders and data folders that may have images as data
        if (file_content.name.lower().startswith(".")) or (file_content.name.lower() in ["data", "images", "dataset"]):
            continue
        # IGNORE files that don't have '.' in their names
        elif (file_content.type == 'file') and (
                '.' not in str(file_content.path)):  # len(file_content.name.split(".")) == 1:
            continue
        # CHECK for directories and file contents
        elif file_content.type == "dir" or file_content.content is None:
            contents.extend(repo_detail.get_contents(file_content.path))
        else:
            # CHECK for specific files
            if (file_extension not in file_extensions_list) or (file_content.encoding == 'none') or (
                    file_content.encoding is None):
                continue
            else:
                try:
                    cnt_ += 1
                    # decode the file content into a readable format
                    file_content_decoded = file_content.decoded_content.decode("utf-8")

                    # In case of .ipynb file extra processing is required
                    if file_extension == 'ipynb':
                        file_content_decoded = fetch_ipynb_content(file_content_decoded)
                        if file_content_decoded is None:
                            continue

                    # Get the analysed data, complexity score and Halstead report
                    analyzed_data, complexity_score, halsteid_report = analysze_data(file_content_decoded)

                    FILE_DETAILS = {'loc': analyzed_data.loc, 'lloc': analyzed_data.lloc,
                                    'sloc': analyzed_data.sloc, 'comments': analyzed_data.comments,
                                    'single_comments': analyzed_data.single_comments,
                                    'multiline_comments': analyzed_data.multi, 'blank_lines': analyzed_data.blank,
                                    'cyclomatic_complexity': complexity_score, 'halstead_report': halsteid_report}

                    repo_raw_text[file_content.name] = FILE_DETAILS
                except (AssertionError, UnicodeDecodeError, nbformat.reader.NotJSONError, json.decoder.JSONDecodeError):
                    continue

    return repo_raw_text, cnt_




#################################################################################

def get_complexity_score_for_each_repo(scores):
    prompt = PromptTemplate(
        input_variables=['scores'],
        template='''Calculate the complexity of the repository on the basis of the dictionary passed below.
        The dictionary will have the Name of the files present in this whole repo as its keys and the corresponding 
        values to the keys is the score calculated for that file.
        '''
    )


def get_complexity_score_for_each_file(raw_code):
    prompt = PromptTemplate(
        input_variables=['raw_code'],
        template='''Calculate the complexity(cyclomatic complexity) of code snippet given below and 
        return the complexity in float value(round to 2 decimal places) without any explanation. Consider the 
        following criteria while scoring: algorithmic efficiency, readability, maintainability, lines of code, 
        packages and modules imported in the project. 
        code snippet : \n`{raw_code}`''')
    llm = OpenAI(temperature=0.8)
    chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

    value = chain.run(raw_code)
    value = value.replace("\n", "")
    # print(value)
    return value


def process_chunk(text):
    print("CHECK - 1")
    chunks = get_text_chunks(text)
    print(f'no of chunks: {len(chunks)}')
    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_texts(chunks, embeddings)
    # Create a question-answering chain using the index
    # print("CHECK - 2")
    context = """You are Super smart Github Repository AI system. You are a super intelligent AI that answers questions 
    about Github Repositories and can understand the technical complexity if the repo.

    You are:
        - helpful & friendly
        - good at answering complex questions in simple language
        - an expert in all programming languages
        - able to infer the intent of the user's question


    Remember You are an intelligent DICTIONARY Agent who can  understand DICTIONARY data and their contents. 
    You are given a dictionary with keys as the repository names and the values having sub-keys as file names 
    whose values are the file data. You are asked to find the most technically complex and challenging repository 
    from the given dictionary. 
    The data for each repo has been saved in the form of a dictionary like :
    '''data = {'repo1': {'file1': '...', 
                        'file2': '...',
                        #...},
                'repo2': {'file1': '...',
                        'file2': '...',
                        # ...},
                # ...}'''
    This data is embedded and stored using FAISS. Retrieve the data from there and return the repository name that you
    consider to be the most complex.

    To measure the technical complexity of a GitHub repository, You will analyze various factors.
    Additionally, you will consider the programming languages used, the size of the codebase.
    You will Analyze the following GitHub repository factors to determine the technical complexity of the codebase
     and calculate a complexity score for each project:

    1.No. of Files in the repository
    2.languages used in the repository file data
    3.Contents of the repository files
    4.Complexity of the statements used as code
    5.Packages and Modules imported inside the repo
    6.No. of Lines Of logical code

    You can consider other factors as well if you think they are relevant for determining the technical complexity 
    of a GitHub repository.
    Calculate the complexity score for each repo by assigning weights to each factor and summing up the weighted scores. 

    The repo with the highest complexity score will be considered the most technically complex.

    Here is the approach or chain-of-thought process , you can use to reach to the solution :
    Step 1: Analyze each file data and it's contents in the dictionary , each outer key represents a Github Repository
    and inner key represents a file name



        """
    question = "Which is the most complex repo ? Why?[answer in less than 100 words]"
    prompt_template = f"""

        Understand the following to answer the question in an efficient way

        {context}

        Question: {question}
        Now answer the question. Let's think step by step:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context"]
    )
    print("CHECK - 3")
    chain_type_kwargs = {"prompt": PROMPT}

    chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectors.as_retriever(),
        input_key="question",
        chain_type_kwargs=chain_type_kwargs
    )
    print("Most Technically Complex Github Repository is")

    result = chain.run()
    # docs = knowledge_base.similarity_search_with_score(prompt)
    # llm = OpenAI()
    # chain = load_qa_chain(llm, chain_type="stuff")
    # with get_openai_callback() as cb:
    #     response = chain.run(input_documents=docs, question=user_question)
    #     print(cb)
    return result
