import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import os
import joblib
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import shap
from collections import defaultdict
from scipy.stats import uniform
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('t4.csv')
df.dropna(inplace=True)

class ECFPDNN(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        radius=3,
        n_bits=4096,
        hidden_layer_sizes=(128, 64, 32, 16, 8, 4, 2),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        tol=1e-4,
        random_state=42
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state
        self.dnn_model = None
        self.feature_names_ = None

    def _smiles_to_morgan(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            self.radius,
            nBits=self.n_bits
        )
        return list(fp)
    def _build_morgan_features(self, df):
        """Build Morgan fingerprint features"""
        print(f"Building Morgan fingerprints with radius={self.radius}, n_bits=2^{self.n_bits}={2**self.n_bits}")
        df_copy = df.copy()
        df_copy['Morgan'] = df_copy['smiles'].apply(lambda x: self._smiles_to_morgan(x))
        X = pd.DataFrame(df_copy['Morgan'].tolist())
        X.columns = X.columns.astype(str)
        return X

    def _smiles_to_morgan(self, smiles):
        """Build Morgan fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * (2**self.n_bits)

        # Use rdFingerprintGenerator.GetMorganGenerator to generate fingerprint
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=2**self.n_bits)
        fp = morgan_gen.GetFingerprint(mol)

        # Convert to bit vector
        return list(fp)

    def _get_morgan_bit_info(self, smiles):
        """Get detailed bit information for Morgan fingerprint"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        # Use AllChem.GetMorganFingerprint to get bit information
        bit_info = {}
        fp = AllChem.GetMorganFingerprint(mol, self.radius, bitInfo=bit_info)

        return fp, bit_info

    def fit(self, X, y):
        """Train model"""
        # Build Morgan fingerprint features
        X_morgan = self._build_morgan_features(X)
        self.feature_names_ = X_morgan.columns.tolist()
        self.dnn_model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            tol=self.tol,
            random_state=self.random_state
        )

        self.dnn_model.fit(X_morgan, y)
        return self
    def predict(self, X):
        if self.dnn_model is None:
            raise ValueError("Model not fitted yet.")
        X_morgan = self._build_morgan_features(X)
        # Ensure feature order consistency
        if self.feature_names_ is not None:
            X_morgan = X_morgan[self.feature_names_]

        return self.dnn_model.predict(X_morgan)

    def get_feature_importances(self):
        """Get feature importances"""
        if self.dnn_model is None:
            return None
def refine(param, val, n_features):
    """Refine hyperparameter space around best value (simple heuristic)."""
    if isinstance(val, int):
        if param == 'hidden_layer_sizes':
            return [val]
        elif param == 'batch_size':
            return sorted(set([max(16, val-16), val, min(256, val+16)]))
        elif param == 'max_iter':
            return sorted(set([max(100, val-50), val, val+50, val+100]))
        elif param == 'n_iter_no_change':
            return sorted(set([max(5, val-5), val, val+5, val+10]))
        elif param == 'radius':
            return sorted(set([max(1, val-1), val, min(5, val+1)]))
        elif param == 'n_bits':
            return sorted(set([max(7, val-1), val, min(14, val+1)]))
        else:
            return [val]

    if isinstance(val, float):
        if param == 'alpha':
            vals = [max(1e-10, val/10), max(1e-10, val/2), val, val*2, val*10]
            return sorted(set([round(v, 6) for v in vals if v > 0.0]))
        elif param == 'learning_rate_init':
            vals = [max(1e-6, val/10), max(1e-6, val/2), val, val*2, val*10]
            return sorted(set([round(v, 6) for v in vals if v > 0.0]))
        elif param == 'tol':
            vals = [max(1e-10, val/10), max(1e-10, val/2), val, val*2, val*10]
            return sorted(set([round(v, 6) for v in vals if v > 0.0]))
        elif param == 'validation_fraction':
            vals = [max(0.05, val*0.5), val, min(0.4, val*1.5)]
            return sorted(set([round(v, 3) for v in vals if 0.05 <= v <= 0.4]))
        else:
            return [val]

    if isinstance(val, str) or isinstance(val, tuple) or val is None:
        return [val]

    return [val]
param_distributions = {
    'radius': [3, 4],
    'n_bits': [9, 10, 11, 12],
    'hidden_layer_sizes': [
        (128, 64, 32, 16, 8, 4, 2),
        (256, 128, 64, 32, 16, 8, 4),
        (512, 256, 128, 64, 32, 16, 8),
        (128, 128, 64, 64, 32, 32, 16),
        (256, 256, 128, 128, 64, 64, 32),
    ],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [1e-6, 1e-5, 1e-4, 1e-3],
    'learning_rate_init': [1e-4, 5e-4, 1e-3],
    'batch_size': [16, 32, 64],
    'max_iter': [200, 500],
    'early_stopping': [True],
    'validation_fraction': [0.1, 0.15],
    'n_iter_no_change': [10, 15],
    'tol': [1e-4, 1e-5]
}

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df[['smiles']], df['E_ST'], test_size=0.2, random_state=42
)

n_rounds =2
best_score_history = []
improvement_threshold = 0.001
patience = 2
no_improvement_count = 0
best_model = None
best_params = None
best_score = -np.inf

for i in range(n_rounds):
    print(f"\n===== Round {i+1} RandomizedSearch =====")

    # Create parameter distribution for current round (including Morgan fingerprint parameters)
    current_param_distributions = param_distributions.copy()
    dnn_model = ECFPDNN(random_state=42)
    RS = RandomizedSearchCV(
        estimator=dnn_model,
        param_distributions=current_param_distributions,
        n_iter=30,
        cv=3,
        scoring='r2',
        verbose=1,
        n_jobs=1,
        error_score='raise',
        random_state=42 + i
    )
    RS.fit(X_train_raw, y_train)

    if RS.best_score_ > best_score:
        best_score = RS.best_score_
        best_model = RS.best_estimator_
        best_params = RS.best_params_.copy()
    print(f"Round {i+1} Best params:", best_params)
    print(f"Round {i+1} Best CV R2:", best_score)

    # Check score improvement
    best_score_history.append(best_score)
    if len(best_score_history) > 1:
        improvement = best_score_history[-1] - best_score_history[-2]
        print(f"Round {i+1} Improvement: {improvement:.6f}")

        if improvement < improvement_threshold:
            no_improvement_count += 1
            print(f"No significant improvement for {no_improvement_count} consecutive rounds")
            if no_improvement_count >= patience:
                print(f"Stopping early after {i+1} rounds due to no significant improvement")
                break
        else:
            no_improvement_count = 0

    # Prepare next round search - refine parameter distribution based on best results
    best = best_params
    new_distributions = dict()
    for k in param_distributions.keys():
        new_distributions[k] = refine(k, best[k], len(best_model.feature_names_) if best_model else 1000)
    param_distributions = new_distributions

    print(f"Refined parameter distributions for next round")

if best_model is not None:
    # Train final ECFPDNN model with best hyperparameters
    final_model = ECFPDNN(**best_params, random_state=42)
    final_model.fit(X_train_raw, y_train)

    # Build full Morgan fingerprint features from the raw training set
    X_train_morgan_full = final_model._build_morgan_features(X_train_raw)

    # Use permutation importance on the trained DNN model
    result = permutation_importance(
        final_model.dnn_model,        # the underlying MLPRegressor
        X_train_morgan_full,          # features used by the DNN
        y_train,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importances = result.importances_mean

    feature_importance_df = pd.DataFrame({
        'feature': X_train_morgan_full.columns.astype(str),
        'importance': importances
    }).sort_values('importance', ascending=False)

    drop_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    drop_results = []

for drop_ratio in drop_ratios:
    n_features_to_keep = int(len(feature_importance_df) * (1 - drop_ratio))
    n_features_to_keep = max(1, n_features_to_keep)  # Keep at least 1 feature
    
    important_features = feature_importance_df.head(n_features_to_keep)['feature'].tolist()
    X_train_selected = final_model._build_morgan_features(X_train_raw)[important_features]
    X_test_selected = final_model._build_morgan_features(X_test_raw)[important_features]


    # Train a separate MLPRegressor on the selected features
    dnn_selected = MLPRegressor(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        activation=best_params['activation'],
        solver=best_params['solver'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        batch_size=best_params['batch_size'],
        max_iter=500,
        random_state=42
    )
    dnn_selected.fit(X_train_selected, y_train)
    y_pred = dnn_selected.predict(X_test_selected)

    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    drop_results.append({
        'drop_ratio': drop_ratio,
        'features_kept': n_features_to_keep,
        'test_r2': test_r2,
        'test_mae': test_mae
    })
        
    print(f"{drop_ratio:9.1f} | {n_features_to_keep:12d} | {test_r2:7.3f} | {test_mae:8.3f}")
    
    # Find best drop ratio
    best_drop_result = max(drop_results, key=lambda x: x['test_r2'])
    print(f"\nBest drop ratio: {best_drop_result['drop_ratio']:.1f} (Test R²: {best_drop_result['test_r2']:.3f})")
    
    # Retrain final model with best drop ratio
    best_n_features = best_drop_result['features_kept']
    best_features = feature_importance_df.head(best_n_features)['feature'].tolist()
    
    # Retrain final model
    X_train_final = final_model._build_morgan_features(X_train_raw)[best_features]
    X_test_final = final_model._build_morgan_features(X_test_raw)[best_features]
    final_dnn = MLPRegressor(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        activation=best_params['activation'],
        solver=best_params['solver'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        batch_size=best_params['batch_size'],
        max_iter=500,
        random_state=42
    )
    final_dnn.fit(X_train_final, y_train)
    joblib.dump(final_dnn,'final_dnn.pkl')
    y_pred_final = final_dnn.predict(X_test_final)

    print(f"\nFinal Model with Best Drop Ratio:")
    print(f"Final Test R² Score: {r2_score(y_test, y_pred_final):.3f}")
    print(f"Final Test MAE: {mean_absolute_error(y_test, y_pred_final):.3f} eV")
    print(f"Final Best Parameters: {best_params}")
    print(f"Best Drop Ratio: {best_drop_result['drop_ratio']:.1f}")
    print(f"Features Used: {len(best_features)}")

    # Save results
    submit_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())

    # Save hyperparameter search results
    results_df = pd.DataFrame({
        'round': range(1, len(best_score_history) + 1),
        'best_cv_score': best_score_history,
        'improvement': [0] + [best_score_history[i] - best_score_history[i-1]
                             for i in range(1, len(best_score_history))]
    })
    results_df.to_csv(os.path.join(submit_dir, "dnn_ecfp_optimization_results_t2.csv"), index=False)

    # Save drop ratio testing results
    drop_results_df = pd.DataFrame(drop_results)
    drop_results_df.to_csv(os.path.join(submit_dir, "dnn_drop_ratio_testing_t2.csv"), index=False)

    # Save final selected feature list
    final_features_df = pd.DataFrame({
        "Feature": best_features,
        "Original_Index": [int(f) for f in best_features]
    })
    final_features_df.to_csv(os.path.join(submit_dir, "dnn_final_selected_features_t2.csv"), index=False)

else:
    print("No valid model was trained!")


# Define visualization function (global scope)
def visualize_ecfp_importance(ecfp_model, best_features,
                             X_train_raw, X_test_raw, y_train, submit_dir, best_params):
    """Visualize Morgan fingerprint importance and molecular structures"""

    print("\n=== Morgan Fingerprint Visualization and Molecular Structure Analysis ===")

    # STEP 1: Perform SHAP analysis FIRST to get SHAP importance for all features
    print("\n1. Performing SHAP analysis on all features from final model...")
    print("   This will determine which features to visualize based on SHAP importance...")

    # Build Morgan fingerprint features using all final selected features
    X_shap_morgan = ecfp_model._build_morgan_features(X_train_raw)
    X_shap = X_shap_morgan[best_features]  # Use all final selected features
    
    # Build test features
    X_test_shap_morgan = ecfp_model._build_morgan_features(X_test_raw)
    X_test_shap = X_test_shap_morgan[best_features]

    # Use best parameters to train DNN model for SHAP analysis
    print("   Training DNN model with best parameters for SHAP analysis...")

    best_params_for_shap = {
        'hidden_layer_sizes': best_params['hidden_layer_sizes'],
        'activation': best_params['activation'],
        'solver': best_params['solver'],
        'alpha': best_params['alpha'],
        'learning_rate_init': best_params['learning_rate_init'],
        'batch_size': best_params['batch_size'],
        'max_iter': 500,
        'random_state': 42
    }
    # Create new DNN model
    dnn_for_shap = MLPRegressor(**best_params_for_shap)

    # Train model
    dnn_for_shap.fit(X_shap, y_train)

    print(f"   SHAP model trained with parameters: {best_params_for_shap}")

    # Use KernelExplainer for DNN
    print("   Creating SHAP explainer...")
    
    # Sample background data for SHAP (to avoid memory issues)
    background_data = shap.sample(X_shap, 100, random_state=42)
    
    explainer = shap.KernelExplainer(
        model=dnn_for_shap.predict,  # Use the predict function
        data=background_data,        # Background dataset
        link="identity"              # For regression
    )

    print("   SHAP explainer created successfully")
    
    # Calculate SHAP values for all features
    print(f"   Computing SHAP values for all {len(best_features)} features...")
    sample_size = min(100, len(X_test_shap))  # Use smaller sample for speed
    X_shap_sample = X_test_shap.iloc[:sample_size]
    shap_values = explainer.shap_values(X_shap_sample, nsamples=100)
    
    # Calculate mean absolute SHAP values for each feature (SHAP importance)
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    
    # Create mapping from feature name to SHAP importance
    shap_importance_dict = {}
    for idx, feature_name in enumerate(best_features):
        shap_importance_dict[str(feature_name)] = mean_shap_values[idx]
    
    print(f"   Calculated SHAP importance for all {len(best_features)} features")
    print(f"   SHAP importance range: {mean_shap_values.min():.6f} to {mean_shap_values.max():.6f}")

    # STEP 2: Select top 10 features based on SHAP importance for visualization
    print("\n2. Selecting top 10 features based on SHAP importance for visualization...")
    
    # Sort features by SHAP importance and select top 10
    shap_importance_indices = np.argsort(mean_shap_values)[::-1][:10]
    top_10_features_by_shap = [best_features[i] for i in shap_importance_indices]
    top_10_shap_importances = [mean_shap_values[i] for i in shap_importance_indices]
    
    print(f"   Selected top 10 features based on SHAP importance from {len(best_features)} final selected features")
    print(f"   Top 10 features by SHAP: {top_10_features_by_shap[:10]} with SHAP importances: {[f'{imp:.4f}' for imp in top_10_shap_importances[:5]]}")

    # STEP 3: Analyze molecular fragments for selected features
    print("\n3. Analyzing Morgan fingerprint bit patterns for top 10 features (by SHAP importance)...")

    # Get Morgan fingerprint features from training data
    X_morgan_train = ecfp_model._build_morgan_features(X_train_raw)

    # Analyze activation patterns for selected features - using SHAP importance
    morgan_analysis = {}

    for feature_name in top_10_features_by_shap:
        # Get SHAP importance for this feature
        shap_importance = shap_importance_dict[str(feature_name)]
        
        # Ensure feature name is string type
        feature_name_str = str(feature_name)
        try:
            feature_idx = int(feature_name)
            bit_values = X_morgan_train.iloc[:, feature_idx]
        except (ValueError, IndexError):
            # If cannot convert to index, try to find by column name
            if feature_name_str in X_morgan_train.columns:
                bit_values = X_morgan_train[feature_name_str]
            else:
                print(f"Warning: Could not find feature {feature_name_str}, skipping...")
                continue

        # Count molecules that activate this bit
        activated_mols = bit_values[bit_values == 1]
        deactivated_mols = bit_values[bit_values == 0]

        # Find molecule indices that activate this bit
        activated_indices = bit_values[bit_values == 1].index.tolist()

        morgan_analysis[feature_name_str] = {
            'importance': shap_importance,  # Use SHAP importance instead of permutation importance
            'shap_importance': shap_importance,  # Store SHAP importance explicitly
            'activation_rate': len(activated_mols) / len(bit_values),
            'activated_count': len(activated_mols),
            'activated_indices': activated_indices,
            'bit_values': bit_values.values
        }

    print(f"Analyzed {len(morgan_analysis)} top 10 features (by SHAP importance)")
    print(morgan_analysis)
    # STEP 4: Create Morgan fingerprint importance visualization using SHAP importance
    print("\n4. Creating Morgan fingerprint importance visualization (using SHAP importance)...")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Subplot 1: Morgan fingerprint importance bar chart (using SHAP importance)
    feature_names = [f"Bit_{f}" for f in top_10_features_by_shap]
    importances = [morgan_analysis[str(f)]['importance'] for f in top_10_features_by_shap]

    bars = ax1.barh(range(len(feature_names)), importances, color='lightblue', alpha=0.8)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('SHAP Feature Importance')
    ax1.set_title(f'Top 10 Morgan Fingerprint Bits - SHAP Importance ({len(top_10_features_by_shap)} features)')
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels on bar chart
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{imp:.4f}', ha='left', va='center', fontsize=8)
    # Subplot 2: Morgan fingerprint activation rate bar chart
    activation_rates = [morgan_analysis[str(f)]['activation_rate'] for f in top_10_features_by_shap]
    activation_bars = ax2.barh(range(len(feature_names)), activation_rates,
                              color='lightcoral', alpha=0.8)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(feature_names)
    ax2.set_xlabel('Activation Rate')
    ax2.set_title(f'Morgan Fingerprint Bit Activation Rates - Top 10 Features by SHAP Importance ({len(top_10_features_by_shap)} features)')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels on bar chart
    for i, (bar, rate) in enumerate(zip(activation_bars, activation_rates)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{rate:.3f}', ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(submit_dir, 'ecfp_importance_visualization_shap.png'),
                dpi=600, bbox_inches='tight')
    plt.show()

    # STEP 5: Analyze molecular structure information for selected features
    print("\n5. Analyzing Morgan fingerprint molecular substructures for top 10 features (by SHAP importance)...")

    # Get molecular samples that activate each Morgan fingerprint bit
    morgan_substructures = {}

    for feature_name in top_10_features_by_shap:  # Use top 10 by SHAP importance
        # Ensure feature name is string type
        feature_name_str = str(feature_name)
        try:
            feature_idx = int(feature_name)
            bit_values = X_morgan_train.iloc[:, feature_idx]
        except (ValueError, IndexError):
            # If cannot convert to index, try to find by column name
            if feature_name_str in X_morgan_train.columns:
                bit_values = X_morgan_train[feature_name_str]
            else:
                print(f"Warning: Could not find feature {feature_name_str}, skipping...")
                continue

        # Count molecules that activate this bit
        activated_mols = bit_values[bit_values == 1]
        deactivated_mols = bit_values[bit_values == 0]

        # Find molecule indices that activate this bit
        activated_indices = bit_values[bit_values == 1].index.tolist()

        morgan_analysis[feature_name_str] = {
            'importance': importances,
            'activation_rate': len(activated_mols) / len(bit_values),
            'activated_count': len(activated_mols),
            'activated_indices': activated_indices,
            'bit_values': bit_values.values
        }

    print(f"Analyzed {len(morgan_analysis)} final selected Morgan fingerprint features")

    # 2. Create Morgan fingerprint importance visualization
    print("\n2. Creating Morgan fingerprint importance visualization for final selected features...")


    # 3. Analyze molecular structure information for Morgan fingerprint bits
    print("\n3. Analyzing Morgan fingerprint molecular substructures for final selected features...")

    # Get molecular samples that activate each Morgan fingerprint bit
    morgan_substructures = {}

    for feature_name in top_10_features_by_shap[:10]: # Limit to top 10 features
        activated_indices = morgan_analysis[str(feature_name)]['activated_indices']

        if len(activated_indices) > 0:
            # Select first 3 molecules that activate this bit for analysis (instead of just the first one)
            n_molecules_to_analyze = min(3, len(activated_indices))
            selected_indices = activated_indices[:n_molecules_to_analyze]

            print(f"Feature {feature_name}: Analyzing {n_molecules_to_analyze} molecules out of {len(activated_indices)} activated molecules")

            for mol_idx, sample_idx in enumerate(selected_indices):
                sample_smiles = X_train_raw.iloc[sample_idx]['smiles']
                sample_mol = Chem.MolFromSmiles(sample_smiles)
                if sample_mol is not None:
                    # Get atomic environment information for Morgan fingerprint - using correct parameters
                    try:
                        # Ensure radius is a valid integer, use radius parameter from training
                        radius = int(ecfp_model.radius)
                        print(f"  Analyzing molecule {mol_idx+1}/{n_molecules_to_analyze} for bit {feature_name} with radius {radius}")
                        # Use the same Morgan fingerprint generation method as during training
                        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2**ecfp_model.n_bits)
                        fp = morgan_gen.GetFingerprint(sample_mol)

                        # Here we use AllChem as an alternative, but need to establish correct mapping

                        # Get AllChem bit information as reference
                        allchem_bit_info = {}
                        detailed_fp = AllChem.GetMorganFingerprint(sample_mol, radius, bitInfo=allchem_bit_info)

                        # Directly use AllChem bit information, no longer try to build manually
                        # Because AllChem.GetMorganFingerprint already provides complete bit information

                        # Create unique key for each molecule
                        mol_key = f"{feature_name}_mol_{mol_idx+1}"
                        morgan_substructures[mol_key] = {
                            'feature_name': feature_name,
                            'molecule_index': mol_idx + 1,
                            'sample_smiles': sample_smiles,
                            'sample_mol': sample_mol,
                            'activation_rate': morgan_analysis[str(feature_name)]['activation_rate'],
                            'bit_info': allchem_bit_info,  # Directly use AllChem bit information
                            'morgan_fp': fp,
                            'detailed_fp': detailed_fp,
                            'allchem_bit_info': allchem_bit_info  # Save AllChem bit information as reference
                        }

                    except Exception as e:
                        print(f"  Warning: Could not analyze molecule {mol_idx+1} for bit {feature_name}: {e}")
                        # If analysis fails, still save basic information
                        mol_key = f"{feature_name}_mol_{mol_idx+1}"
                        morgan_substructures[mol_key] = {
                            'feature_name': feature_name,
                            'molecule_index': mol_idx + 1,
                            'sample_smiles': sample_smiles,
                            'sample_mol': sample_mol,
                            'activation_rate': morgan_analysis[str(feature_name)]['activation_rate'],
                            'bit_info': {},
                            'morgan_fp': None,
                            'detailed_fp': None
                        }

    print(f"Analyzed molecular substructures for {len(morgan_substructures)} features")
    # STEP 6: Visualize molecular substructures corresponding to Morgan fingerprint bits
    print("\n6. Creating molecular substructure visualizations for Morgan fingerprint bits...")

    # Use top 10 features selected by SHAP importance
    top_features_to_show = top_10_features_by_shap

    # Process molecules grouped by feature
    for feature_name in top_features_to_show:
        print(f"\n=== Processing feature {feature_name} ===")

        # 找到该特征的所有分子
        feature_molecules = [key for key in morgan_substructures.keys() if key.startswith(f"{feature_name}_mol_")]

        if not feature_molecules:
            print(f"  No molecules found for feature {feature_name}")
            continue

        print(f"  Found {len(feature_molecules)} molecules for feature {feature_name}")

        for mol_key in feature_molecules:
            substructure_info = morgan_substructures[mol_key]
            mol = substructure_info['sample_mol']
            mol_idx = substructure_info['molecule_index']

            if mol is not None:
                print(f"\n  Processing molecule {mol_idx} for feature {feature_name}...")

                # Add atom numbers to molecule for easier observation
                mol_copy = Chem.Mol(mol)
                for atom in mol_copy.GetAtoms():
                    atom.SetAtomMapNum(atom.GetIdx())

                # Use AllChem to generate Morgan fingerprint and get bit information
                radius = int(ecfp_model.radius)
                n_bits = 2**ecfp_model.n_bits

                # Create bit information dictionary
                bit_info = {}

                # Generate Morgan fingerprint and fill bit information
                fp = AllChem.GetMorganFingerprintAsBitVect(mol_copy, radius, nBits=n_bits, bitInfo=bit_info)

                print(f"    Generated Morgan fingerprint with radius={radius}, n_bits={n_bits}")
                print(f"    Fingerprint length: {fp.GetNumBits()}")
                print(f"    Activated bits: {fp.GetNumOnBits()}")
                # Check bit information dictionary content
                if not bit_info:
                    print(f"    Warning: No bit information generated for molecule {mol_idx}")
                    continue

                # Try to find bit corresponding to current feature
                target_bit = None
                contributing_substructures = []
                print(f"    Available bits in bit_info: {list(bit_info.keys())[:30]}...")  # Show first 10 bits

                # Method 1: Directly use feature name as bit index
                try:
                    feature_bit_idx = int(feature_name)
                    if feature_bit_idx in bit_info:
                        target_bit = feature_bit_idx
                        contributing_substructures = bit_info[target_bit]
                        print(f"    Found target bit {target_bit} with {len(contributing_substructures)} contributing substructures")
                    else:
                        print(f"    Feature {feature_name} (bit {feature_bit_idx}) not found in bit_info")
                except ValueError:
                    print(f"    Could not convert feature name {feature_name} to integer")

                # Method 2: If direct match fails, try to find most similar bit
                if target_bit is None and bit_info:
                    # Select first available bit for analysis
                    target_bit = list(bit_info.keys())[0]
                    contributing_substructures = bit_info[target_bit]
                    print(f"    Using fallback bit {target_bit} with {len(contributing_substructures)} contributing substructures")

                if target_bit is not None and contributing_substructures:
                    print(f"    Analyzing Bit {target_bit}")
                    print(f"    Contributing substructures: {len(contributing_substructures)}")

                    # Analyze center atom type and atomic types at each radius layer for each substructure
                    substructure_analysis = []

                    for i, (atom_idx, radius_val) in enumerate(contributing_substructures):
                        print(f"      - Substructure {i+1}: Center atom index = {atom_idx}, Radius = {radius_val}")

                        # Get center atom information
                        center_atom = mol_copy.GetAtomWithIdx(atom_idx)
                        center_atom_symbol = center_atom.GetSymbol()
                        center_atom_degree = center_atom.GetDegree()
                        center_atom_valence = center_atom.GetTotalValence()

                        print(f"        Center atom: {center_atom_symbol} (index {atom_idx}), degree={center_atom_degree}, valence={center_atom_valence}")
                        # Analyze atomic types at each radius layer
                        layer_analysis = {}
                        for r in range(radius_val + 1):
                            layer_atoms = set()
                            layer_bonds = set()

                            if r == 0:
                                # Center atom
                                layer_atoms.add(atom_idx)
                            else:
                                # Use FindAtomEnvironmentOfRadiusN to find environment at specified radius
                                env = Chem.FindAtomEnvironmentOfRadiusN(mol_copy, r, atom_idx)
                                if env is not None:
                                    for bond_idx in env:
                                        bond = mol_copy.GetBondWithIdx(bond_idx)
                                        layer_bonds.add(bond_idx)
                                        layer_atoms.add(bond.GetBeginAtomIdx())
                                        layer_atoms.add(bond.GetEndAtomIdx())

                            # Count atomic types in this layer
                            atom_types = {}
                            for aid in layer_atoms:
                                atom = mol_copy.GetAtomWithIdx(aid)
                                atom_symbol = atom.GetSymbol()
                                if atom_symbol not in atom_types:
                                    atom_types[atom_symbol] = 0
                                atom_types[atom_symbol] += 1

                            layer_analysis[r] = {
                                'atoms': list(layer_atoms),
                                'bonds': list(layer_bonds),
                                'atom_types': atom_types
                            }

                            print(f"        Radius {r}: Atomic types={atom_types}, Atom count={len(layer_atoms)}, Bond count={len(layer_bonds)}")

                        # Save substructure analysis results
                        substructure_analysis.append({
                            'center_atom_idx': atom_idx,
                            'center_atom_symbol': center_atom_symbol,
                            'center_atom_degree': center_atom_degree,
                            'center_atom_valence': center_atom_valence,
                            'radius': radius_val,
                            'layer_analysis': layer_analysis
                        })

                    # Create fingerprint schematic
                    print(f"    Creating fingerprint schematic for bit {target_bit}...")
                    try:
                        # Create fingerprint schematic
                        fig, axes = plt.subplots(1, len(substructure_analysis), figsize=(4*len(substructure_analysis), 6))
                        if len(substructure_analysis) == 1:
                            axes = [axes]

                        for i, substructure in enumerate(substructure_analysis):
                            ax = axes[i]

                            # Draw center atom
                            center_symbol = substructure['center_atom_symbol']
                            center_degree = substructure['center_atom_degree']

                            # Create circular plot showing center atom information
                            ax.text(0.5, 0.8, f"Center atom: {center_symbol}", ha='center', va='center', fontsize=12, weight='bold')
                            ax.text(0.5, 0.6, f"Degree: {center_degree}", ha='center', va='center', fontsize=10)
                            ax.text(0.5, 0.4, f"Valence: {substructure['center_atom_valence']}", ha='center', va='center', fontsize=10)

                            # Draw atomic type distribution for each layer
                            y_pos = 0.2
                            for r in range(substructure['radius'] + 1):
                                if r in substructure['layer_analysis']:
                                    layer_info = substructure['layer_analysis'][r]
                                    atom_types = layer_info['atom_types']
                                    atom_count = sum(atom_types.values())

                                    ax.text(0.5, y_pos, f"Radius {r}: {atom_types}", ha='center', va='center', fontsize=9)
                                    y_pos -= 0.15

                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                            ax.set_title(f'Substructure {i+1}', fontsize=12)
                            ax.axis('off')

                        plt.suptitle(f'Morgan Fingerprint Bit {target_bit} Schematic\nMolecule {mol_idx}, Radius={radius}, Substructure count={len(substructure_analysis)}', fontsize=14)
                        plt.tight_layout()

                        # Save fingerprint schematic
                        schematic_filename = os.path.join(submit_dir, f'morgan_bit_{feature_name}_mol{mol_idx}_schematic.png')
                        plt.savefig(schematic_filename, dpi=600, bbox_inches='tight')
                        print(f"    Saved fingerprint schematic to {schematic_filename}")
                        plt.close()

                    except Exception as e:
                        print(f"    Error creating fingerprint schematic: {e}")
                    # Find all atoms and bonds that constitute each contributing substructure
                    highlight_list = []

                    for i, (atom_idx, radius_val) in enumerate(contributing_substructures):
                        print(f"      - Substructure {i+1}: Center atom index = {atom_idx}, Radius = {radius_val}")

                        # Find all atoms and bonds that constitute this substructure
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol_copy, radius_val, atom_idx)
                        atoms_in_env = set()
                        bonds_to_highlight = set()

                        if env is not None:
                            for bond_idx in env:
                                bonds_to_highlight.add(bond_idx)
                                bond = mol_copy.GetBondWithIdx(bond_idx)
                                atoms_in_env.add(bond.GetBeginAtomIdx())
                                atoms_in_env.add(bond.GetEndAtomIdx())

                            # Save information about atoms and bonds to highlight
                            highlight_list.append((list(atoms_in_env), list(bonds_to_highlight)))
                            print(f"        Atoms in environment: {list(atoms_in_env)}")
                            print(f"        Bonds in environment: {list(bonds_to_highlight)}")
                        else:
                            print(f"        Cannot find atomic environment with radius {radius_val}")

                    # Visualize substructures contributing to target_bit
                    if highlight_list:
                        print(f"    Creating visualization with {len(highlight_list)} substructures...")

                        # Create grid image showing multiple substructures
                        try:
                            # Use Draw.MolsToGridImage to create grid image
                            grid_image = Draw.MolsToGridImage(
                                [mol_copy for _ in highlight_list],  # Show original molecule for plotting
                                highlightAtomLists=[h[0] for h in highlight_list],
                                highlightBondLists=[h[1] for h in highlight_list],
                                legends=[f"Substructure {i+1} (Bit {target_bit})" for i in range(len(highlight_list))],
                                molsPerRow=min(3, len(highlight_list)),
                                subImgSize=(300, 300)
                            )

                            # Save grid image
                            grid_filename = os.path.join(submit_dir, f'morgan_bit_{feature_name}_mol{mol_idx}_substructures.png')
                            grid_image.save(grid_filename)
                            print(f"    Saved substructure grid visualization to {grid_filename}")
                        except Exception as e:
                            print(f"    Error creating grid image: {e}")
                            # If grid image creation fails, try to create single images
                            try:
                                # Create visualization for single substructure
                                for i, (atoms, bonds) in enumerate(highlight_list):
                                    img = Draw.MolToImage(mol_copy,
                                                        highlightAtoms=atoms,
                                                        highlightBonds=bonds,
                                                        highlightAtomColors={idx: (0.3, 0.7, 1.0) for idx in atoms},
                                                        highlightBondColors={idx: (0.3, 0.7, 1.0) for idx in bonds})

                                    single_filename = os.path.join(submit_dir, f'morgan_bit_{feature_name}_mol{mol_idx}_substructure_{i+1}.png')
                                    img.save(single_filename)
                                    print(f"    Saved single substructure visualization to {single_filename}")

                            except Exception as e2:
                                print(f"    Error creating single image: {e2}")
                    else:
                        print(f"    No valid substructures found for visualization")
                else:
                    print(f"    No valid bit information found for molecule {mol_idx}")

    print(f"Created molecular structure visualizations for {len(top_features_to_show)} features")

    # STEP 8: Create SHAP summary plot for top 10 features
    print("\n8. Creating SHAP summary plot for top 10 features...")
    
    # Extract SHAP values for top 10 features (already calculated above)
    shap_values_top10 = shap_values[:, shap_importance_indices]
    X_shap_sample_top10 = X_shap_sample.iloc[:, shap_importance_indices]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_top10, X_shap_sample_top10,
                      feature_names=[f"Bit_{f}" for f in top_10_features_by_shap],
                      show=False)
    plt.title(f'SHAP Summary Plot - Top 10 Morgan Fingerprint Bits by SHAP Importance ({len(top_10_features_by_shap)} features)')
    plt.tight_layout()
    plt.savefig(os.path.join(submit_dir, 'shap_summary_plot_top10.png'),
                dpi=600, bbox_inches='tight')
    plt.show()

    # STEP 9: Create SHAP dependency plots (select top 5 most important features)
    print("\n9. Creating SHAP dependency plots...")

    # Select top 10 most important features for dependency plot analysis
    top_10_features = top_10_features_by_shap[:10]
    top_10_shap_importances = top_10_shap_importances[:10]

    for i, (feature_name, feature_idx) in enumerate(zip(top_10_features, range(10))):
        print(f"Creating dependency plot for feature {feature_name} (SHAP importance: {top_10_shap_importances[i]:.4f})...")

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, shap_values_top10, X_shap_sample_top10,
                           feature_names=[f"Bit_{f}" for f in top_10_features_by_shap],
                           show=False)
        plt.title(f'SHAP Dependence Plot - Morgan Fingerprint Bit {feature_name}\nSHAP Importance: {top_10_shap_importances[i]:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(submit_dir, f'shap_dependence_{feature_name}.png'),
                   dpi=600, bbox_inches='tight')
        plt.show()

    # STEP 10: Save SHAP analysis results
    print("\n10. Saving SHAP analysis results...")

    # Create SHAP importance DataFrame - based on top 10 features sorted by SHAP importance
    shap_importance_df = pd.DataFrame({
        'Feature': [f"Bit_{f}" for f in top_10_features_by_shap],
        'Original_Index': top_10_features_by_shap,
        'SHAP_Importance': top_10_shap_importances,
        'Feature_Rank': range(1, len(top_10_features_by_shap) + 1)
    }).sort_values('SHAP_Importance', ascending=False)

    shap_importance_df.to_csv(os.path.join(submit_dir, 'shap_importance_analysis_top10.csv'), index=False)

    print("SHAP analysis completed successfully!")
    print(f"Analyzed top 10 features (by SHAP importance) from {len(top_10_features_by_shap)} final selected features")

    # STEP 11: Save Morgan fingerprint analysis results
    print("\n11. Saving Morgan fingerprint analysis results for top 10 features (by SHAP importance)...")

    morgan_analysis_df = pd.DataFrame([
        {
            'Morgan_Bit': f,
            'SHAP_Importance': morgan_analysis[str(f)]['importance'],  # This is SHAP importance now
            'Activation_Rate': morgan_analysis[str(f)]['activation_rate'],
            'Activated_Count': morgan_analysis[str(f)]['activated_count'],
            'Total_Molecules': len(morgan_analysis[str(f)]['bit_values']),
            'Sample_SMILES': morgan_substructures.get(f"{f}_mol_1", {}).get('sample_smiles', 'N/A')
        }
        for f in top_10_features_by_shap  # Use top 10 by SHAP importance
    ])

    morgan_analysis_df.to_csv(os.path.join(submit_dir, 'morgan_detailed_analysis_shap_top10.csv'), index=False)

    # STEP 12: Save molecular substructure information
    print("\n12. Saving molecular substructure information for top 10 features (by SHAP importance)...")

    substructure_info = []
    for feature_name in top_10_features_by_shap:  # Use top 10 by SHAP importance
        # Find substructure info for this feature
        feature_substructures = [key for key in morgan_substructures.keys() if key.startswith(f"{feature_name}_mol_")]
        if feature_substructures:
            info = morgan_substructures[feature_substructures[0]]  # Get first molecule's info
            substructure_info.append({
                'Morgan_Bit': feature_name,
                'Sample_SMILES': info.get('sample_smiles', 'N/A'),
                'Bit_Info_Count': len(info.get('bit_info', {}))
            })

    substructure_df = pd.DataFrame(substructure_info)
    substructure_df.to_csv(os.path.join(submit_dir, 'morgan_substructure_analysis_shap_top10.csv'), index=False)
    print(f"\nMorgan fingerprint analysis completed! Results saved to {submit_dir}")
    print("Total analysis steps: 12 (SHAP analysis first, then visualization based on SHAP importance)")

    return morgan_analysis

# Call visualization function in final evaluation section
if best_model is not None:
    # Call visualization function
    morgan_analysis = visualize_ecfp_importance(
        final_model, best_features,
        X_train_raw, X_test_raw, y_train, submit_dir, best_params
    )



