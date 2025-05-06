from collections import defaultdict
import pandas as pd

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
        'Environmental Science': ['environmental', 'ecology', 'climate']
    }
    
    description = description.lower()
    matched_categories = []
    for category, keywords in categories.items():
        if any(keyword in description for keyword in keywords):
            matched_categories.append(category)
    
    return matched_categories if matched_categories else ['Other']

categorized_packages = defaultdict(list)

for _, row in df.iterrows():
    package = row['X1']
    description = str(row['X2'])  # Convert to string to handle potential NaN values
    categories = categorize_package(description)
    for category in categories:
        categorized_packages[category].append((package, description))

# Prepare data for export
export_data = []
for category, packages in categorized_packages.items():
    for package, description in packages:
        export_data.append({'Category': category, 'Package': package, 'Description': description})

# Create a DataFrame
export_df = pd.DataFrame(export_data)

# Save to CSV
export_df.to_csv('cran_packages_by_area_multi.csv', index=False)

print('Updated spreadsheet created: cran_packages_by_area_multi.csv')

# Print summary
category_counts = export_df['Category'].value_counts()
print("\
Package counts by category:")
print(category_counts)

print(f"\
Total unique packages: {export_df['Package'].nunique()}")
print(f"Total package-category pairs: {len(export_df)}")

# Calculate the frequency of each category excluding 'Other'
category_counts_filtered = df_filtered['Category'].value_counts()

# Plot the bar chart
plt.figure(figsize=(12, 8))
category_counts_filtered.plot(kind='bar')
plt.title('Frequency of CRAN Packages by Category (Excluding Other)')
plt.xlabel('Category')
plt.ylabel('Number of Packages')
plt.xticks(rotation=45)
plt.show()
