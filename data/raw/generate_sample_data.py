import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'column_a': np.random.randn(n_samples),
    'column_b': ['value_' + str(i % 3) for i in range(n_samples)],
    'target': np.random.randn(n_samples)
})

# Save to CSV
data.to_csv('data/raw/data.csv', index=False) 