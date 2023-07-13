import re
import nbformat
from langchain import PromptTemplate, OpenAI, LLMChain, FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from nbconvert import PythonExporter
from langchain.text_splitter import RecursiveCharacterTextSplitter


def fetch_ipynb_content(content):
    try:
        notebook = nbformat.reads(content, as_version=4)
    except:
        return None

    python_exporter = PythonExporter()
    python_code, _ = python_exporter.from_notebook_node(notebook)
    return python_code

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        # language=Language,
        add_start_index=True
    )
    return text_splitter.split_text(text)


def extract_modules(data, file_format):
    modules_each_file = set()
    pattern_set = {".py": r'^\s*(?:import\s+(\w+|\w+\.\w+)\b|from\s+(\w+|\w+\.\w+)\s+import\s+([^\n]+))',
                   ".ipynb": r'^\s*(?:import\s+(\w+|\w+\.\w+)\b|from\s+(\w+|\w+\.\w+)\s+import\s+([^\n]+))',
                   ".cpp": r'#include\s+["<]([\w\.]+)[">]',
                   ".c": r'#include\s+["<]([\w\.]+)[">]',
                   ".java": r'import\s+([\w.]+);',
                   ".js": r'(?:import\s+[^;]*?\s+from\s+)?[\'"]([^"\']+)["\']'}
    pattern = pattern_set[file_format]
    import_statements = re.findall(pattern, data, flags=re.MULTILINE)
    # print(import_statements)
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


def calculate_metrics(code):
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
    print("CHECK - 2")
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
