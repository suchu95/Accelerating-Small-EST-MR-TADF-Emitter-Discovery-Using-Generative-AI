import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import uniform
import warnings
import os

warnings.filterwarnings('ignore')

seed = 42
np.random.seed(seed)
print("Loading data and generating ECFP fingerprints...")

df = pd.read_csv('t4.csv')
df = df.dropna(subset=['smiles', 'E_ST']).copy()

def smiles_to_ecfp_safe(smiles, radius=3, n_bits=4096):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return list(fp)
N_BITS = 4096
df['ECFP'] = df['smiles'].apply(lambda s: smiles_to_ecfp_safe(s, radius=3, n_bits=N_BITS))

df = df[df['ECFP'].notnull()].reset_index(drop=True)

X = np.array(df['ECFP'].tolist(), dtype=np.float32)
y = df['E_ST'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=seed
)

print(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

print("\nSetting up hyperparameter search...")

mlp = MLPRegressor(
    activation='relu',
    solver='adam',
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    tol=1e-4,
    random_state=seed
)

param_distributions = {
    'hidden_layer_sizes': [
        (128, 64, 32, 16, 8, 4, 2),         # 7 layers (descending)
        (256, 128, 64, 32, 16, 8, 4),       # 7 layers (descending)
        (512, 256, 128, 64, 32, 16, 8),     # 7 layers (descending)
        (1024, 512, 256, 128, 64, 32, 16),  # 7 layers (descending)
        (128, 128, 64, 64, 32, 32, 16),     # 7 layers (repeated sizes)
        (256, 256, 128, 128, 64, 64, 32),   # 7 layers (repeated sizes)
        (512, 512, 256, 256, 128, 128, 64), # 7 layers (repeated sizes)
        (128, 256, 128, 64, 128, 64, 32),   # 7 layers (alternating)
        (256, 512, 256, 128, 256, 128, 64), # 7 layers (alternating)
    ],
    'alpha': uniform(1e-6, 1e-3),
    'learning_rate_init': [1e-4, 5e-4, 1e-3, 2e-3],
    'batch_size': [16, 32, 64],
    'activation': ['relu', 'tanh']
}

random_search = RandomizedSearchCV(
    estimator=mlp,
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='r2',
    verbose=2,
    random_state=seed,
    n_jobs=-1
)

print("Starting randomized search... (This may take several minutes)")
random_search.fit(X_train, y_train)

print("\n--- Hyperparameter Tuning Complete ---")
print(f"Best cross-validated R² score: {random_search.best_score_:.4f}")
print("Best parameters found:")
for param, value in random_search.best_params_.items():
    print(f"  - {param}: {value}")

print("\nEvaluating the best model on the held-out test set...")

best_mlp = random_search.best_estimator_

y_pred = best_mlp.predict(X_test)

final_r2 = r2_score(y_test, y_pred)
final_mae = mean_absolute_error(y_test, y_pred)

# Print the requested metrics
print(f"R² Score: {final_r2:.4f}")
print(f"MAE: {final_mae:.4f} eV")


