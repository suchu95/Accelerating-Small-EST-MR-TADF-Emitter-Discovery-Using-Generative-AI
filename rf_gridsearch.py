import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

# === Load dataset ===
df = pd.read_csv('t4.csv')
df.dropna(subset=["smiles", "E_ST"], inplace=True)

def smiles_to_ecfp(smiles, radius=3, n_bits=4096):
    mol = Chem.MolFromSmiles(smiles)
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits))

df['ECFP'] = df['smiles'].apply(smiles_to_ecfp)
X = pd.DataFrame(df['ECFP'].tolist())
y = df['E_ST']
X.columns = X.columns.astype(str)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Refine helper ===
def refine(param, val):
    if isinstance(val, int):
        if param == 'min_samples_split':
            return sorted(set([max(2, val-2), max(2, val-1), val, val+1, val+2]))
        elif param == 'min_samples_leaf':
            return sorted(set([max(1, val-2), max(1, val-1), val, val+1, val+2]))
        elif param in ['n_estimators', 'max_depth']:
            return sorted(set([max(1, val-2), max(1, val-1), val, val+1, val+2]))
        else:
            return [val]
    if isinstance(val, float):
        if param == 'max_features':
            vals = [min(1.0, max(0.01, val/2)), min(1.0, max(0.01, val*0.8)),
                    min(1.0, max(0.01, val)), min(1.0, max(0.01, val*1.2)),
                    min(1.0, max(0.01, val*2))]
            return sorted(set([round(v, 4) for v in vals if 0.0 < v <= 1.0]))
        else:
            vals = [max(1e-10, val/2), max(1e-10, val*0.8),
                    val, val*1.2, val*2]
            return sorted(set([round(v, 6) for v in vals if v > 0.0]))
    if isinstance(val, str):
        return [val]
    if val is None:
        return [None]
    return [val]

# === Initial grid for RandomForestRegressor ===
param_grid = {
    'n_estimators': [50, 100, 150, 200, 250],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5],
    'bootstrap': [True, False],
    'ccp_alpha': [0.0, 0.001, 0.01]
}

# === Iterative search ===
n_rounds = 5
best_model = None
best_params_final = None
final_r2 = None
final_mae = None

for i in range(n_rounds):
    print(f"\n===== Round {i+1} GridSearch =====")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    GS = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        refit=True,
        verbose=0,
        n_jobs=-1
    )
    GS.fit(X_train, y_train)

    # Best params from this round
    best_params = GS.best_params_

    # Train with best params
    model_round = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    model_round.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_round = model_round.predict(X_test)
    r2_round = r2_score(y_test, y_pred_round)
    mae_round = mean_absolute_error(y_test, y_pred_round)

    print(f"Test R² (Round {i+1}): {r2_round:.3f}")
    print(f"Test MAE (Round {i+1}): {mae_round:.3f} eV")
    print(f"Best parameters (Round {i+1}): {best_params}")

    # Save final model
    best_model = model_round
    best_params_final = best_params
    final_r2 = r2_round
    final_mae = mae_round

    # Prepare narrower search for next round
    param_grid = {k: refine(k, best_params[k]) for k in param_grid.keys()}

# === Final summary ===
print("\n===== Final Model Evaluation =====")
print(f"Final Test R² Score: {final_r2:.3f}")
print(f"Final Test MAE: {final_mae:.3f} eV")
print(f"Parameters leading to final result: {best_params_final}")

# === Save final search results ===
submit_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())
pd.DataFrame(GS.cv_results_).sort_values("rank_test_score").to_csv(
    os.path.join(submit_dir, "rf_iterative_gridsearch_results_t2.csv"),
    index=False
)
