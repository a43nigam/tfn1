import pandas as pd

n_rows = 100

data = {
    'text': [f'This is a dummy book excerpt number {i}. Once upon a time...' for i in range(n_rows)]
}
df = pd.DataFrame(data)
df.to_csv('pg19.csv', index=False)
print('Dummy PG-19 CSV generated: pg19.csv') 