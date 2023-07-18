# importing Libraries
import subprocess

# Define the command to install dependencies
install_command = ['pip', 'install', '-r', 'requirements.txt']

# Run the command to install the dependencies
subprocess.check_call(install_command)

import os
import time
import github
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from utilities import *


# Defining the main function
def main(username, git_access_token):
    """Takes 2 parameter as input, username and github_access_token;
    returns the saved data into a csv file"""
    if username:
        try:
            # STEP - 1 : FETCHING THE REPO NAMES
            x = time.time()
            user, repo_names, repo_descriptions, repo_urls = display_repo_names_and_url(username, git_access_token)
            y = time.time()
            st.write(f"Repo Names Has been fetched . Time taken for fetching repo names : {y - x} sec")
            print('#' * 50)
            all_repo_details = {}
            cnt = 0

            # STEP - 2 : FETCHING THE RAW DATA FROM EACH REPO
            st.write("Proceeding to fetch raw text.")
            for repo in repo_names:
                cnt += 1
                raw_text, total_files = get_each_repo_data(user, repo)
                if total_files == 0 or total_files is None:
                    continue
                if raw_text:
                    all_repo_details[repo] = raw_text
                else:
                    continue
            st.write("Data Fetching Process Completed.")

            # STEP - 3 : CONVERT THE DATA INTO A STRUCTURED FORMAT
            st.write("Proceeding to convert fetched data into a csv file...")
            x = time.time()
            df = pd.DataFrame(columns=["repo", "file","url", "loc", "lloc", "sloc", "blank_lines", "comments",
                                       "multiline_comments", "single_comments", "cyclomatic_complexity",
                                       "halstead_report"])
            data_rows = []
            index_value = -1
            for rep, files in all_repo_details.items():
                index_value += 1
                desription = repo_descriptions.get(f'{rep}', '')
                repo_url = repo_urls[index_value]
                for file, file_info in files.items():
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
                        "description":desription,
                        "url": repo_url,
                        "file": file,
                        "loc": loc,
                        "lloc": lloc,
                        "sloc": sloc,
                        "comments": comments,
                        "blank_lines": blank_lines,
                        "multiline_comments": multiline_comments,
                        "single_comments": single_comments,
                        "cyclomatic_complexity": cyclomatic_complexity,
                        "halstead_report": halstead_report
                    })
            df = pd.concat([df, pd.DataFrame(data_rows)], ignore_index=True)
            # saving the dataframe into a csv
            df.to_csv("repo_details.csv")
            y = time.time()
            st.write(f"CSV file saved. Time taken {y - x}")

        except github.RateLimitExceededException:
            print('Rate Limit exceeded...wait for an hour and try again...')
    # else:
    #     print('Invalid Username')


if __name__ == "__main__":
    # load the dotenv for accessing the api keys
    load_dotenv()
    git_access_token = os.getenv("GITHUB_API_KEY")  # GITHUB Api Key
    # openai_api_key = os.getenv("OPENAI_API_KEY")  # OpenAI Api Key

    st.title('Github Automated Analysis | Most Technically Complex And Challenging Repo')
    github_url = st.text_input("Please Enter The GitHub User's URL or Username For Analysis")
    # git_access_token = st.text_input("Please provide the Github API key", type="password")
    openai_api_key = st.text_input("Please provide the OpenAI API key", type="password")
    if len(github_url.split('/')) > 1:
        username = github_url.split('/')[-1]
    else:
        username = github_url

    # Running the code
    with st.form('Github_repo_analysis', clear_on_submit=True):
        submitted = st.form_submit_button('Search')
        if submitted and openai_api_key.startswith('sk-'):
            with st.spinner('Calculating...'):
                # main function to process the data... STEP - 1, STEP - 2, STEP - 3
                main(username, git_access_token)

                # STEP - 4 : CALCULATING THE MOST COMPLEX REPO THROUGH OPENAI
                st.write("Data Fetched....Proceeding to calculation...")
                response = get_results(openai_api_key)  # function to get the results from OpenAI
                st.write(response)
    #####################################################################################
