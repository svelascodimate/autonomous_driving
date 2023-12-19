# This file contains code for extracting bug reports for SDVs
# Part of the research on software bugs in SDVs
# Date: 18 December 2023
# Author: Leyli (Aya) Garryyeva

#from google.colab import drive
#drive.mount('/content/drive')

#!pip install requests

# Import Libraries

import requests
import json
from urllib.parse import urlencode
from datetime import datetime, timedelta
import pandas as pd

#####
# Create the github mining function, get the data, and save it as json file

git_orgs = ["eclipse-leda", "eclipse-sumo", "eclipse-zenoh",
            "eclipse-ecal", "eclipse-velocitas", "eclipse-tractusx",
             "eclipse-ankaios", "adore", "eclipse-bluechi",
            "eclipse-chariott",  "eclipse-kuksa", "eclipse-muto",
             "eclipse-uprotocol"
            # "adaaa", "cloe", "sommr", "openmcx",
            ] # list of orgs working on SDVs

github_token = "" # input your token here

def get_github_issues(org, repo, label="bug", state="open", token=None):
    base_url = "https://api.github.com"
    endpoint = f"/repos/{org}/{repo}/issues"
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    params = {
        "labels": label,
        "state": state,
    }

    response = requests.get(base_url + endpoint, params=params, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: Unable to fetch issues from {org}/{repo}")
        return None

def save_to_file(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    select_bugs = []

    for org in git_orgs:
        print(f"\nFetching bug-labeled issues for organization: {org}")
        repos_url = f"https://api.github.com/orgs/{org}/repos"
        repos_response = requests.get(repos_url, headers={"Authorization": f"Bearer {github_token}"})

        if repos_response.status_code == 200:
            repos_data = repos_response.json()

            for repo in repos_data:
                repo_name = repo["name"]
                issues_open = get_github_issues(org, repo_name, label="bug", state="open", token=github_token)
                issues_closed = get_github_issues(org, repo_name, label="bug", state="closed", token=github_token)

                if issues_open:
                    select_bugs.extend(
                        {"org": org, "repo": repo_name, "issue": issue} for issue in issues_open
                    )
                if issues_closed:
                    select_bugs.extend(
                        {"org": org, "repo": repo_name, "issue": issue} for issue in issues_closed
                    )

        else:
            print(f"Error: Unable to fetch repositories for organization {org}")

    return select_bugs

# subset the data with select features
def create_bug_subset(select_bugs):
    bug_subset = []
    for bug in select_bugs:
        org = bug['org']
        repo = bug['repo']
        state = bug['issue']['state']
        issue_url = bug['issue']['html_url']
        issue_title = bug['issue']['title']
        issue_description = bug['issue']['body'] if bug['issue']['body'] else "No description available"

        bug_subset.append({
            "organization": org,
            "repository": repo,
            "state": state,
            "issue_url": issue_url,
            "issue_title": issue_title,
            "issue_description": issue_description
        })

    return bug_subset


if __name__ == "__main__":
    select_bugs = main()
    bug_subset = create_bug_subset(select_bugs)

    # Save the data to a file
    save_to_file(bug_subset, "select_bugs.json") # save_data_to_json(extracted_data, OUTPUT_PATH + query + '.json')
    print("\nSelected Bug-labeled Issues saved to select_bugs.json")


# open the json file and start working with the data
#import json

def load_from_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    # Load data from the file
    select_bugs = load_from_file("select_bugs.json")

    print(bug_subset)


# EDA
# Convert the bug subset to a pandas DataFrame
bug_df = pd.DataFrame(bug_subset)

# Display basic information about the dataset
print("Basic Information about the Bug Subset:")
print(bug_df.info())

# Display summary statistics for categorical columns
print("\nSummary Statistics for Categorical Columns:")
print(bug_df.describe(include='object'))

# Display the first few rows of the DataFrame
print("\nFirst Few Rows of the Bug Subset:")
print(bug_df.head())

# Display unique values and counts for categorical columns
print("\nUnique Values and Counts for Categorical Columns:")
for column in bug_df.select_dtypes(include='object').columns:
    print(f"\n{column}:")
    print(bug_df[column].value_counts())

# Display word count statistics for issue descriptions
bug_df['issue_description_word_count'] = bug_df['issue_description'].apply(lambda x: len(str(x).split()))
print("\nWord Count Statistics for Issue Descriptions:")
print(bug_df['issue_description_word_count'].describe())

#    print(bug_df.describe())


# Retrieve the bug data to Excel file
from google.colab import files
# Save bug_df as an Excel file
bug_df.to_excel("bug_df.xlsx")
# Download bug_df.xlsx to local computer
files.download("bug_df.xlsx")

### Clustering 
# 1. Cluster by issue title only

# Install necessary libraries if not already installed
# !pip install pandas scikit-learn

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Extract the relevant column for clustering
documents = bug_df['issue_title']

# Use TF-IDF Vectorizer to convert text data to numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Specify the number of clusters (up to 10 in this case)
num_clusters = 10

# Perform KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(X)

# Add the cluster assignments to the DataFrame
bug_df['cluster_title'] = cluster_assignments

# Apply PCA to reduce the feature space to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Add the PCA components to the DataFrame
bug_df['PCA1'] = X_pca[:, 0]
bug_df['PCA2'] = X_pca[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 8))
for cluster_id in range(num_clusters):
    cluster_data = bug_df[bug_df['cluster_title'] == cluster_id]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster_id}')

plt.title('K-means Clustering of Bug Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()


# 2. Cluster by issue description only

# Install necessary libraries if not already installed
# !pip install pandas scikit-learn

# Extract the relevant column for clustering
documents = bug_df['issue_description']

# Use TF-IDF Vectorizer to convert text data to numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Specify the number of clusters (up to 10 in this case)
num_clusters = 10

# Perform KMeans clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(X)

# Add the cluster assignments to the DataFrame
bug_df['cluster_description'] = cluster_assignments

# Apply PCA to reduce the feature space to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Add the PCA components to the DataFrame
bug_df['PCA1'] = X_pca[:, 0]
bug_df['PCA2'] = X_pca[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 8))
for cluster_id in range(num_clusters):
    cluster_data = bug_df[bug_df['cluster_description'] == cluster_id]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster_id}')

plt.title('K-means Clustering of Bug Data')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Retrieve the bug and cluster data to Excel file

from google.colab import files
# Save bug_df as an Excel file
bug_df.to_excel("bug_df_clusters.xlsx")
# Download bug_df.xlsx to local computer
files.download("bug_df_clusters.xlsx")






