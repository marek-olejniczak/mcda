import os
import pandas as pd

if __name__ == "__main__":
    if not os.path.exists("hospital_data.csv"):
        raise FileNotFoundError("hospital_data.csv not found. Please run preprocess.py first.")
    df = pd.read_csv("hospital_data.csv")
    
    # map 1,2 stars to 0, and 3,4,5 stars to 1
    df["hospital_overall_rating"] = df["hospital_overall_rating"].apply(lambda x: 0 if x in [1, 2] else 1)
    df.to_csv("binarized_hospital_data.csv", index=False)
    print("Binarized hospital data saved to binarized_hospital_data.csv")