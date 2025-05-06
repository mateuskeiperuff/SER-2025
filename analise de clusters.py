import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Read the Excel file
df = pd.read_excel('base de dados.xlsx')

# Clean the Downloads column by removing commas and converting to numeric
df['Downloads'] = df['Downloads'].str.replace(',', '').astype(float)

# Create TF-IDF vectors from descriptions
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description'])

# Apply K-means clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Category'] = kmeans.fit_predict(tfidf_matrix)

# Get the most common terms for each cluster
feature_names = vectorizer.get_feature_names_out()
cluster_terms = {}
for i in range(n_clusters):
    center = kmeans.cluster_centers_[i]
    top_terms_idx = center.argsort()[-5:][::-1]
    top_terms = [feature_names[idx] for idx in top_terms_idx]
    cluster_terms[i] = top_terms
    print(f"\
Cluster {i} main themes: {', '.join(top_terms)}")
    print("Example packages:")
    print(df[df['Category'] == i][['Package Name', 'description']].head(3))

print("\
Total packages per cluster:")
print(df['Category'].value_counts())

# Increase the number of clusters to better separate distinct purposes
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Category'] = kmeans.fit_predict(tfidf_matrix)

# Get the most common terms for each cluster
feature_names = vectorizer.get_feature_names_out()
for i in range(n_clusters):
    center = kmeans.cluster_centers_[i]
    top_terms_idx = center.argsort()[-5:][::-1]
    top_terms = [feature_names[idx] for idx in top_terms_idx]
    print(f"\
Cluster {i} main themes: {', '.join(top_terms)}")
    print("Top packages:")
    cluster_df = df[df['Category'] == i].sort_values('Downloads', ascending=False)
    print(cluster_df[['Package Name', 'Downloads', 'description']].head(3))

print("\
Packages per cluster:")
print(df['Category'].value_counts().sort_index())

# Group the data by cluster and display the first few entries of each group
grouped_df = df.groupby('Category')

# Display the first few entries of each cluster group
for name, group in grouped_df:
    print(f"\
Cluster {name}:")
    print(group[['Package Name', 'Downloads', 'description']].head())

print("Data grouped by cluster.")

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate total downloads per cluster
downloads_per_cluster = df.groupby('Category')['Downloads'].sum().reset_index()

# Plotting
plt.figure(figsize=(12, 6))
sns.barplot(x='Category', y='Downloads', data=downloads_per_cluster, palette='viridis')
plt.title('Total Downloads per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Total Downloads')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate key statistics and top packages for each cluster
for cluster in sorted(df['Category'].unique()):
    cluster_data = df[df['Category'] == cluster]
    top_packages = cluster_data.nlargest(3, 'Downloads')
    total_downloads = cluster_data['Downloads'].sum()
    
    print(f"\
Cluster {cluster}:")
    print(f"Total Downloads: {total_downloads:,.0f}")
    print("Top Packages:")
    for _, row in top_packages.iterrows():
        print(f"- {row['Package Name']}: {row['description'][:100]}...")
        

# Plotting with full numbers on the y-axis and labels for each bar
plt.figure(figsize=(12, 6))
barplot = sns.barplot(x='Category', y='Downloads', data=downloads_per_cluster, palette='viridis')

# Add labels to each bar
for index, row in downloads_per_cluster.iterrows():
    barplot.text(index, row['Downloads'], f'{row['Downloads']:,}', color='black', ha="center")

plt.title('Total Downloads per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Total Downloads')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np

# Group packages by category and get key information
for cat in range(20):  # 0-19 clusters
    cluster_data = df[df['Category'] == cat]
    if len(cluster_data) > 0:
        print(f"\
Cluster {cat}")
        print("Number of packages:", len(cluster_data))
        print("Top 3 packages by downloads:")
        top3 = cluster_data.nlargest(3, 'Downloads')[['Package Name', 'description', 'Downloads']]
        for _, row in top3.iterrows():
            print(f"- {row['Package Name']}: {row['Downloads']:,} downloads")
            print(f"  Description: {row['description'][:100]}...")
            

# Creating a box plot to visualize the distribution of downloads across clusters
plt.figure(figsize=(14, 7))
sns.boxplot(x='Category', y='Downloads', data=df, palette='coolwarm')
plt.title('Distribution of Downloads Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Downloads')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Creating a box plot to visualize the distribution of downloads across clusters
plt.figure(figsize=(14, 7))
sns.boxplot(x='Category', y='Downloads', data=df, palette='coolwarm')
plt.title('Distribution of Downloads Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Downloads')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Creating violin plots for download distributions by cluster
plt.figure(figsize=(15, 8))
sns.violinplot(x='Category', y='Downloads', data=df, palette='coolwarm')
plt.title('Download Distribution Shapes Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Downloads')
plt.yscale('log')  # Using log scale for better visibility
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Calculate the number of packages in each cluster
cluster_sizes = df['Category'].value_counts().sort_index()

# Plotting the cluster sizes
plt.figure(figsize=(12, 6))
cluster_sizes.plot(kind='bar', color='lightseagreen')
plt.title('Number of Packages in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Packages')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Calculate the number of packages in each cluster
cluster_sizes = df['Category'].value_counts().sort_index()

# Plotting the cluster sizes
plt.figure(figsize=(12, 6))
cluster_sizes.plot(kind='bar', color='lightseagreen')
plt.title('Number of Packages in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Packages')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# Get top 20 packages for each cluster
top_packages = []
for cluster in df['Category'].unique():
    cluster_data = df[df['Category'] == cluster]
    top_20 = cluster_data.nlargest(20, 'Downloads')[['Category', 'Package Name', 'Downloads', 'description']]
    top_packages.append(top_20)

# Combine all results
result = pd.concat(top_packages)

# Save to Excel with proper formatting
result.to_excel('top_20_packages_per_cluster.xlsx', index=False)

print("Top packages saved to 'top_20_packages_per_cluster.xlsx'")
print("\
First few entries of the data:")
print(result.head())
