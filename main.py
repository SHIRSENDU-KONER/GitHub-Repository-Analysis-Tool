import json
import re
import os
import time

import github
import radon
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import radon.raw
from langchain.chat_models import ChatOpenAI
from radon.complexity import cc_visit
from radon.metrics import h_visit
from github import Github
from utilities import *
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain import FAISS, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

    def __repr__(self):
        attributes = ', '.join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({attributes})"


def analysze_data(data):
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


def display_repo_names_and_url(username, access_token):
    g = Github(access_token)
    user = g.get_user(username)
    cnt = 0
    repo_names = []
    repo_descriptions = {}
    # print('CHECK-2')
    for repo in user.get_repos():
        cnt += 1
        # print(f"{cnt}. Repository: {repo.name}")
        repo_names.append(repo.name)
        description = repo.description
        repo_descriptions[repo.name] = description if description else None
        # break
    # print(f"This User {username} has {cnt} repositories. They are:")

    return user, repo_names, repo_descriptions


def get_each_repo_data(user, repo_name):
    repo_detail = user.get_repo(repo_name)
    cnt_ = 0
    try:
        contents = repo_detail.get_contents('')
        # print(contents)
    except:
        # print("No data")
        return None, None

    global all_file_chunks
    repo_raw_text = {}
    # print("check-1")
    file_extensions_list = ["py", "ipynb", "cpp", "c", "java", "php", 'js']

    while contents:
        file_content = contents.pop(0)
        file_extension = file_content.name.split(".")[-1]
        # print(file_content.path, file_content.type, file_content.name)
        if (file_content.name.lower().startswith(".")) or (file_content.name.lower() in ["data", "images", "dataset"]):
            # print('check')
            continue
        if (file_content.type == 'file') and (
                '.' not in str(file_content.path)):  # len(file_content.name.split(".")) == 1:
            # print(file_content.name)
            # print(contents)
            continue
        elif file_content.type == "dir" or file_content.content is None:
            contents.extend(repo_detail.get_contents(file_content.path))
        else:
            if (file_extension not in file_extensions_list) or (file_content.encoding == 'none') or (
                    file_content.encoding is None):
                continue
            else:
                try:
                    # print("check-2")
                    # file_extension = file_content.name.split(".")[-1]
                    cnt_ += 1
                    # print(cnt_, '|', file_content.name, '|', file_content.encoding, '|', file_content.path)
                    # print(cnt_, file_content.path, file_content.encoding)
                    file_content_decoded = file_content.decoded_content.decode("utf-8")
                    if file_extension == 'ipynb':
                        # print(cnt_, '|', file_content.name, '|', file_content.encoding, '|', file_content.path)
                        # print('#')
                        # print("check-3")
                        file_content_decoded = fetch_ipynb_content(file_content_decoded)
                        if file_content_decoded is None:
                            # print(file_content.name, 'file is INVALID FILE.')
                            continue
                    #     analyzed_data, complexity_score, halsteid_report = analysze_data(file_content_decoded)
                    # else:
                    # print("check-4")
                    analyzed_data, complexity_score, halsteid_report = analysze_data(file_content_decoded)
                    # print("check-5")
                    # print('##')
                    # print(analyzed_data)
                    # print(f"complexity : {complexity_score}")
                    # print(f'Halsteid : {halsteid_report}')
                    # '''
                    # radon.raw.analyse(CODE)->> this returns the followings:
                    #     LOC, LLOC, SLOC, COMMENTS, SINGLE-LINE COMMENTS, MULTILINE COMMENTS, BLANK LINES,
                    # radon.metrics.h_visit(CODE) returns the Halstead Report,
                    # radon.complexity.ComplexityVisitor.from_code(CODE).total_complexity gives the total complexity
                    #                     '''
                    FILE_DETAILS = {'data': file_content_decoded, 'loc': analyzed_data.loc, 'lloc': analyzed_data.lloc,
                                    'sloc': analyzed_data.sloc, 'comments': analyzed_data.comments,
                                    'single_comments': analyzed_data.single_comments,
                                    'multiline_comments': analyzed_data.multi, 'blank_lines': analyzed_data.blank,
                                    'cyclomatic_complexity': complexity_score, 'halstead_report': halsteid_report}
                    # print('###')
                    repo_raw_text[file_content.name] = FILE_DETAILS
                except (AssertionError, UnicodeDecodeError, nbformat.reader.NotJSONError, json.decoder.JSONDecodeError):
                    continue

                # print(file_content.name, file_content.encoding)
    # st.write(f"Total no of file scanned in this repo : {cnt_}")
    return repo_raw_text, cnt_  # total_lines_of_code, modules, languages


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        # language=Language,
        add_start_index=True
    )
    return text_splitter.split_text(text)


def get_results(openai_api_key):
    # print("check-1")
    loader = CSVLoader(file_path='repo_details.csv', encoding="utf-8")
    data = loader.load()
    # print(type(data))
    data = '\n'.join([doc.page_content for doc in data])
    # print(type(data))
    # print("check-2")
    docs = get_text_chunks(data)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    database = FAISS.from_texts(docs, embeddings)
    # print("check-3")
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
        context='''You can consider other factors as well if you think they are relevant for determining the technical complexity of a
        GitHub repository.
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
            - languages used in the repository file data
            - Contents of the repository files
            - Complexity of the statements used as code
            - Packages and Modules imported inside the repo
            - No.of logical lines of code or lines of executable source code
            - Cyclomatic Complexity of the files/codes per repository(to be calculated from the code)
            - Halstead Complexity of the files/codes as per repository(to be calculated from the code)
            - Maintainability Index of the files/codes as per repository(to be calculated from the code)
            Return only the name of the repository, its complexity score and the analysis of the repository showing why 
            it is the most technically challenging/Complex repository. Try to provide a detailed analysis to hold 
            your answer strong within 100 words in a paragraph. The output should be in the following format:
            Repository Name: <name of the repository>
            Complexity Score: <complexity score of the repository>
            Analysis: <analysis of the repository>'''
    result = chain.run(query)
    # print(result)
    print("The Prompt of the Langchain is :", prompt)
    if not result:
        return "No result."
    else:
        return "\n".join(result.split("."))


def main(username, git_access_token):
    if username:
        try:
            x = time.time()
            # print('CHECK-1')
            user, repo_names, repo_descriptions = display_repo_names_and_url(username, git_access_token)
            y = time.time()
            st.write(f"Repo Names Has been fetched . Time taken for fetching repo names : {y - x} sec")
            print('#' * 50)
            all_repo_details = {}
            cnt = 0
            st.write("Proceeding to fetch raw text.")
            for repo in repo_names:
                # st.write(f"Current repo : {repo}")
                cnt += 1
                x = time.time()
                raw_text, total_files = get_each_repo_data(user, repo)
                if total_files == 0 or total_files is None:
                    # print('#' * 50)
                    continue
                y = time.time()
                if raw_text:
                    all_repo_details[repo] = raw_text
                else:
                    continue
                # if cnt == len(repo_names):  # this is for last repo
                    # st.write(f"Data from {repo} repo has been fetched. Time taken : {y - x}")
                    # print(f"Data from {repo} repo has been fetched. Time taken : {y - x}")
                # else:
                    # st.write(f"Data from {repo} repo has been fetched. Time taken : {y - x}."
                    #       f"\n\nProceeding to the next repo if available")
                    # print(f"Data from {repo} repo has been fetched. Time taken : {y - x}."
                    #       f"\n\nProceeding to the next repo if available")
                    # print("*" * 50)
                    # print()
            # print('#' * 50)
            st.write("Data Fetching Process Completed.")
            # print('#' * 50)
            st.write("Proceeding to convert fetched data into a csv file...")
            x = time.time()
            df = pd.DataFrame(columns=["repo", "file", "content", "loc", "lloc", "sloc", "blank_lines", "comments",
                                       "multiline_comments", "single_comments", "cyclomatic_complexity",
                                       "halstead_report"])
            data_rows = []
            for rep, files in all_repo_details.items():
                for file, file_info in files.items():
                    content = str(file_info.get('data', ''))
                    loc = file_info.get('loc', '')
                    lloc = file_info.get('lloc', '')
                    sloc = file_info.get('sloc', '')
                    blank_lines = file_info.get('blank_lines', '')
                    comments = file_info.get('comments', '')
                    multiline_comments = file_info.get('multiline_comments', '')
                    single_comments = file_info.get('single_comments', '')
                    cyclomatic_complexity = file_info.get('cyclomatic_complexity', '')
                    halstead_report = file_info.get('halstead_report', '')

                    # Append a row to the DataFrame
                    data_rows.append({
                        "repo": rep,
                        "file": file,
                        "content": content,
                        "loc": loc,
                        "lloc": lloc,
                        "sloc": sloc,
                        "comments": comments,
                        "blank_lines" : blank_lines,
                        "multiline_comments": multiline_comments,
                        "single_comments": single_comments,
                        "cyclomatic_complexity": cyclomatic_complexity,
                        "halstead_report": halstead_report
                    })
            df = pd.concat([df, pd.DataFrame(data_rows)], ignore_index=True)

            df.to_csv("repo_details.csv")
            y = time.time()
            st.write(f"CSV file saved. Time taken {y - x}")
            # print('#' * 50)
            # st.write("Proceeding for analysis using OpenAI and Langchain.")
            # response = get_results(openai_api_key)
            # result.append(response)
            # st.write(response)
        except github.RateLimitExceededException:
            print('Rate Limit exceeded...wait for an hour and try again...')
    # else:
    #     print('Invalid Username')


if __name__ == "__main__":
    load_dotenv()
    git_access_token = os.getenv("GITHUB_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    # git_access_token = os.getenv("github_api_key_college_acc")

    # username = 'trekhleb'
    st.title('Github Automated Analysis | Most Technically Complex And Challenging Repo')
    github_url = st.text_input("Please Enter The GitHub User's URL or Username For Analysis")
    if len(github_url.split('/')) > 1:
        username = github_url.split('/')[-1]
    else:
        username = github_url
    # username = 'Anivesh-Agnihotri'
    # main(username, git_access_token)
    result = []
    with st.form('myform', clear_on_submit=True):
        # openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
        submitted = st.form_submit_button('Search')
        if submitted and openai_api_key.startswith('sk-'):
            with st.spinner('Calculating...'):
                main(username, git_access_token)
                st.write("Data Fetched....Proceeding to calculation...")
                response = get_results(openai_api_key)
                result.append(response)
                st.write(response)
                # del openai_api_key
    # results = get_results(openai_api_key)
    # print(results)
    #####################################################################################
