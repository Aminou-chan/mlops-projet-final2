from load_data import load_california_housing


df = load_california_housing()
df.to_csv("data/raw/california_housing.csv", index=False)
print("Data saved in data/raw/california_housing.csv")
