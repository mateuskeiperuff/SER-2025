import pandas as pd

# Load the CSV file
df = pd.read_csv('mydata2.csv', encoding='utf-8')

# Remove the header
# Reset the header to the default integer index
df.columns = range(df.shape[1])

# Display the first few rows to confirm changes
print(df.head())

# Reload the CSV file to restore the original headers
df = pd.read_csv('mydata2.csv', encoding='utf-8')

# Display the first few rows to confirm the headers are restored
print(df.head())

# Remove the first row (index 0) from the dataframe
df = df.drop(index=0).reset_index(drop=True)

# Display the first few rows to confirm the change
print(df.head())

# Remove the first column from the dataframe
df = df.drop(columns=df.columns[0])

# Display the first few rows to confirm the change
print(df.head())

from collections import defaultdict
import re

def categorize_package(description):
    categories = {
        'Statistics': ['statistics', 'statistical', 'regression', 'estimation', 'probability'],
        'Machine Learning': ['machine learning', 'predictive', 'classification', 'clustering'],
        'Data Visualization': ['visualization', 'plot', 'graph', 'chart'],
        'Bioinformatics': ['genomic', 'biological', 'gene', 'protein'],
        'Time Series': ['time series', 'temporal', 'forecasting'],
        'Optimization': ['optimization', 'algorithm', 'solver'],
        'Network Analysis': ['network', 'graph theory', 'social network'],
        'Text Mining': ['text', 'natural language', 'nlp', 'sentiment'],
        'Spatial Analysis': ['spatial', 'geographic', 'map', 'gis'],
        'Finance': ['finance', 'economic', 'stock', 'market'],
        'Psychology': ['psychology', 'cognitive', 'behavioral'],
        'Environmental Science': ['environmental', 'ecology', 'climate'],
        'Other': []
    }
    
    description = description.lower()
    for category, keywords in categories.items():
        if any(keyword in description for keyword in keywords):
            return category
    return 'Other'

categorized_packages = defaultdict(list)

for _, row in df.iterrows():
    package = row['X1']
    description = row['X2']
    category = categorize_package(description)
    categorized_packages[category].append((package, description))

# Print the categorized packages
for category, packages in categorized_packages.items():
    print(f"\
{category}:")
    for package, description in packages[:3]:  # Print only the first 3 packages per category
        print(f"  - {package}: {description[:100]}...")  # Truncate description to 100 characters
    if len(packages) > 3:
        print(f"  ... and {len(packages) - 3} more")

print(f"\
Total categories: {len(categorized_packages)}")
print(f"Total packages: {sum(len(packages) for packages in categorized_packages.values())}")

# Fill missing descriptions with an empty string
df['X2'] = df['X2'].fillna('')

# Re-run the categorization process
categorized_packages = defaultdict(list)

for _, row in df.iterrows():
    package = row['X1']
    description = row['X2']
    category = categorize_package(description)
    categorized_packages[category].append((package, description))

# Print the categorized packages
for category, packages in categorized_packages.items():
    print(f"\
{category}:")
    for package, description in packages[:3]:  # Print only the first 3 packages per category
        print(f"  - {package}: {description[:100]}...")  # Truncate description to 100 characters
    if len(packages) > 3:
        print(f"  ... and {len(packages) - 3} more")

print(f"\
Total categories: {len(categorized_packages)}")
print(f"Total packages: {sum(len(packages) for packages in categorized_packages.values())}")
