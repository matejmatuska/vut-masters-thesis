import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet(sys.argv[1])

df = df.groupby(['sample', 'family']).size().reset_index(name='count')

# Step 1: Drop duplicates to get one row per sample
samples = df.drop_duplicates(subset='sample')[['sample', 'family']]

# Step 2: Count number of samples per family
family_counts = samples['family'].value_counts().reset_index()
family_counts.columns = ['family', 'sample_count']

# Step 3: Plot horizontally
sns.barplot(data=family_counts, y='family', x='sample_count')


# Optional: Beautify the plot
plt.title('Number of Occurrences per Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.show()
