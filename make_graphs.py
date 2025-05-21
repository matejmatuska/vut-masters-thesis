import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet(sys.argv[1])

df = df.groupby(['sample', 'family']).size().reset_index(name='count')

samples = df.drop_duplicates(subset='sample')[['sample', 'family']]

family_counts = samples['family'].value_counts().reset_index()
family_counts.columns = ['family', 'sample_count']

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.figure(figsize=(12, 10))
sns.barplot(data=family_counts, y='family', x='sample_count')

# Optional: Beautify the plot
plt.title('Number of Samples per Family')
plt.xlabel('Family')
plt.ylabel('Number of Samples')
plt.tight_layout()
plt.savefig('sample_distrib.pdf', dpi=300)
plt.show()
