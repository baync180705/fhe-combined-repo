import pandas as pd
from sklearn.preprocessing import StandardScaler
import blindml

def main():
    print("1. Loading Pima Indians Diabetes Dataset...")
    # Standard public URL for the Pima dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['preg', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'outcome']
    df = pd.read_csv(url, names=columns)

    # 2. Select Top 2 Features to survive the FHE Noise Budget
    # Glucose and BMI are historically the strongest predictors of diabetes
    X = df[['glucose', 'bmi']].values
    y = df['outcome'].values

    print("2. Scaling data for 6-bit fixed-point FHE (Q6F3)...")
    # Scale the features into the narrow encrypted range used by the current 6-bit regime.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Take a subset for training speed
    # We use a subset so the fully encrypted backward pass finishes in reasonable time.
    X_train = X_scaled[:100].astype("float32").tolist()
    y_train = y[:100].astype("float32").tolist()

    print("3. Loading or generating FHE cryptographic keys (cached on disk when available)...")
    context = blindml.BlindContext.generate()
    model = blindml.BlindLogisticRegression(input_features=2)

    print(f"4. Starting Encrypted Training on {len(X_train)} real-world samples...")
    print("   Debug logging is enabled between epoch summaries.")
    model.fit(
        context,
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        learning_rate=0.1
    )

    print("5. Training complete. Exporting encrypted weights...")
    model.export_model("./")
    print("6. Fhenix deployment files generated successfully.")

if __name__ == "__main__":
    # You may need to run: pip install pandas
    main()
