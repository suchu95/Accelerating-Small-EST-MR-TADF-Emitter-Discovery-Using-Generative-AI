import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem import rdFingerprintGenerator
from xgboost import XGBRegressor                            
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
import os
import joblib
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import shap
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('t4.csv')
df.dropna(inplace=True)

class ECFPXGBoosting(BaseEstimator, RegressorMixin):
    """XGBoost Regressor with Morgan fingerprints."""
    
    def __init__(self, radius=3, n_bits=12, n_estimators=100, max_depth=3, 
                 learning_rate=0.1, subsample=1.0, colsample_bytree=1.0,
                 reg_alpha=0, reg_lambda=1, gamma=0, min_child_weight=1,
                 random_state=42):
        self.radius = radius
        self.n_bits = n_bits  # n_bits=12 means fpSize=4096 (2^12)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.xgb_model = None  # Changed from gbr_model to xgb_model
        self.feature_names_ = None

    def _build_morgan_features(self, df):
        """构建Morgan指纹特征"""
        print(f"Building Morgan fingerprints with radius={self.radius}, n_bits=2^{self.n_bits}={2**self.n_bits}")
        df_copy = df.copy()
        df_copy['Morgan'] = df_copy['smiles'].apply(lambda x: self._smiles_to_morgan(x))
        X = pd.DataFrame(df_copy['Morgan'].tolist())
        X.columns = X.columns.astype(str)
        return X

    def _smiles_to_morgan(self, smiles):
        """构建Morgan指纹"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0] * (2**self.n_bits)

        # 使用rdFingerprintGenerator.GetMorganGenerator生成指纹
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=2**self.n_bits)
        fp = morgan_gen.GetFingerprint(mol)

        # 转换为位向量
        return list(fp)
    def _get_morgan_bit_info(self, smiles):
        """获取Morgan指纹的详细位信息"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        # 使用AllChem.GetMorganFingerprint获取位信息
        bit_info = {}
        fp = AllChem.GetMorganFingerprint(mol, self.radius, bitInfo=bit_info)

        return fp, bit_info

    def fit(self, X, y):
        """训练模型"""
        # 构建Morgan指纹特征
        X_morgan = self._build_morgan_features(X)
        self.feature_names_ = X_morgan.columns.tolist()
        
        # XGBoost参数配置
        self.xgb_model = XGBRegressor(
            learning_rate=self.learning_rate,       # 学习率，控制每个弱学习器的贡献
            n_estimators=self.n_estimators,         # 弱学习器(树)的数量
            max_depth=self.max_depth,               # 每棵树的最大深度
            subsample=self.subsample,               # 训练样本的子采样比例
            colsample_bytree=self.colsample_bytree, # 每棵树列采样的比例
            reg_alpha=self.reg_alpha,               # L1正则化项权重
            reg_lambda=self.reg_lambda,             # L2正则化项权重
            gamma=self.gamma,                       # 节点分裂所需的最小损失减少
            min_child_weight=self.min_child_weight, # 子节点所需的最小样本权重和
            random_state=self.random_state,         # 随机种子，保证结果可复现
            n_jobs=-1                               # 使用所有CPU核心并行处理
        )

        # Train the model
        self.xgb_model.fit(X_morgan, y)
        return self

    def predict(self, X):
        """预测"""
        if self.xgb_model is None:
            raise ValueError("Model not fitted yet")

        X_morgan = self._build_morgan_features(X)
        # 确保特征顺序一致
        if self.feature_names_ is not None:
            X_morgan = X_morgan[self.feature_names_]

        return self.xgb_model.predict(X_morgan)
    def get_feature_importances(self):
        """获取特征重要性"""
        if self.xgb_model is None:  # Changed from gbr_model to xgb_model
            return None
        return self.xgb_model.feature_importances_

def refine(param, val):
    if isinstance(val, int):
        if param == 'min_child_weight':
            return sorted(set([max(1, val-2), max(1, val-1), val, val+1, val+2]))
        elif param in ['n_estimators', 'max_depth']:
            return sorted(set([max(1, val-2), max(1, val-1), val, val+1, val+2]))
        else:
            return [val]
    if isinstance(val, float):
        if param in ['subsample', 'colsample_bytree', 'learning_rate']:
            vals = [min(1, max(0.01, val/2)), min(1, max(0.01, val*0.8)), val,
                    min(1, max(0.01, val*1.2)), min(1, max(0.01, val*2))]
            return sorted(set([round(v, 4) for v in vals if 0.0 < v <= 1.0]))
        elif param in ['reg_alpha', 'reg_lambda', 'gamma']:
            vals = [max(1e-10, val/2), max(1e-10, val*0.8), val, val*1.2, val*2]
            return sorted(set([round(v, 6) for v in vals if v > 0.0]))
        else:
            return [val]
    if isinstance(val, str):
        return [val]
    if val is None:
        return [None]
    return [val]

param_distributions = {
    'radius': [2, 3, 4],
    'n_bits': [9, 10, 11, 12],  # Corresponds to 2^9=512 up to 2^12=4096
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # 学习率，控制每个弱学习器的贡献
    'subsample': [0.7, 0.8, 0.9, 1.0],         # 训练样本的子采样比例
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # **新增**: 每棵树列采样的比例 (NEW: Column sampling ratio per tree)
    'n_estimators': [100, 200, 300, 400, 500], # 弱学习器的数量
    'max_depth': [3, 5, 7, 9],                 # 每棵树的最大深度
    'reg_alpha': [0, 0.1, 0.5, 1.0],           # **新增**: L1正则化项权重 (NEW: L1 regularization)
    'reg_lambda': [0.5, 1.0, 1.5, 2.0],        # **新增**: L2正则化项权重 (NEW: L2 regularization)
    'gamma': [0, 0.1, 0.5, 1.0],               # **新增**: 节点分裂所需的最小损失减少 (NEW: Minimum loss reduction for split)
    'min_child_weight': [1, 3, 5, 7]           # **新增**: 子节点所需的最小样本权重和 (NEW: Minimum sum of instance weight needed in a child)
}

# 训练/测试分割（基于原始数据）
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df[['smiles']], df['E_ST'], test_size=0.2, random_state=42
)

n_rounds = 2
best_score_history = []
improvement_threshold = 0.001
patience = 2
no_improvement_count = 0
best_model = None
best_params = None
best_score = -np.inf

for i in range(n_rounds):
    print(f"\n===== Round {i+1} RandomizedSearch =====")

    # 创建当前轮次的参数分布（包含Morgan指纹参数）
    current_param_distributions = param_distributions.copy()

    # 创建并训练Morgan指纹XGBoost模型
    xgb_model = ECFPXGBoosting(random_state=42)  # Changed from ECFPGradientBoosting to ECFPXGBoosting

    # 使用随机搜索替代网格搜索
    RS = RandomizedSearchCV(
        estimator=xgb_model,  # Changed from gbr to xgb_model
        param_distributions=current_param_distributions,
        n_iter=100,  # 每轮随机搜索20次，大大减少搜索次数
        cv=5,
        scoring='r2',
        verbose=1,
        n_jobs=-1,
        error_score='raise',
        random_state=42 + i  # 每轮使用不同的随机种子
    )

    RS.fit(X_train_raw, y_train)
    
    # 更新最佳模型和参数
    if RS.best_score_ > best_score:
        best_score = RS.best_score_
        best_model = RS.best_estimator_
        best_params = RS.best_params_.copy()

    print(f"Round {i+1} Best params:", best_params)
    print(f"Round {i+1} Best CV R2:", best_score)

    # 检查分数改善情况
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

    # 准备下一轮搜索 - 基于最佳结果精炼参数分布
    best = best_params
    new_distributions = dict()
    for k in param_distributions.keys():
        # 安全地获取参数值，如果参数不存在则使用默认值
        if k in best:
            new_distributions[k] = refine(k, best[k])
        else:
            # 如果参数不在best中，使用原始分布或默认值
            new_distributions[k] = param_distributions[k]

    param_distributions = new_distributions

    print(f"Refined parameter distributions for next round")

# 最终评估
if best_model is not None:
    # 使用最佳参数重新训练最终模型
    final_model = ECFPXGBoosting(**best_params, random_state=42)  # Changed to ECFPXGBoosting
    final_model.fit(X_train_raw, y_train)

    feature_importances = final_model.get_feature_importances()
    
    # Create XGBoost model with best parameters
    xgb_selected = XGBRegressor(
        learning_rate=best_params['learning_rate'],
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        subsample=best_params.get('subsample', 1.0),
        colsample_bytree=best_params.get('colsample_bytree', 1.0),
        reg_alpha=best_params.get('reg_alpha', 0),
        reg_lambda=best_params.get('reg_lambda', 1),
        gamma=best_params.get('gamma', 0),
        min_child_weight=best_params.get('min_child_weight', 1),
        random_state=42,
        n_jobs=-1
    )
    
    xgb_selected.fit(final_model._build_morgan_features(X_train_raw), y_train)
    joblib.dump(xgb_selected, 'final_xgb.pkl')  # Changed filename to final_xgb.pkl

    feature_importance_df = pd.DataFrame({
        'feature': final_model.feature_names_,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)

    print(f"\nTotal features: {len(feature_importance_df)}")
    print(f"Best parameters from randomized search: {best_params}")

    # 测试不同的drop比例
    drop_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    drop_results = []

    print("\nTesting different drop ratios:")
    print("Drop Ratio | Features Kept | Test R² | Test MAE")
    print("-" * 50)

    for drop_ratio in drop_ratios:
        # 计算要保留的特征数量
        n_features_to_keep = int(len(feature_importance_df) * (1 - drop_ratio))
        n_features_to_keep = max(1, n_features_to_keep)  # 至少保留1个特征

        # 选择最重要的特征
        important_features = feature_importance_df.head(n_features_to_keep)['feature'].tolist()

        # 创建只使用重要特征的训练数据
        X_train_selected = final_model._build_morgan_features(X_train_raw)[important_features]
        X_test_selected = final_model._build_morgan_features(X_test_raw)[important_features]

        # Create XGBoost model with best parameters
        xgb_selected = XGBRegressor(
            learning_rate=best_params['learning_rate'],
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            subsample=best_params.get('subsample', 1.0),
            colsample_bytree=best_params.get('colsample_bytree', 1.0),
            reg_alpha=best_params.get('reg_alpha', 0),
            reg_lambda=best_params.get('reg_lambda', 1),
            gamma=best_params.get('gamma', 0),
            min_child_weight=best_params.get('min_child_weight', 1),
            random_state=42,
            n_jobs=-1
        )

        xgb_selected.fit(X_train_selected, y_train)
        y_pred = xgb_selected.predict(X_test_selected)

        test_r2 = r2_score(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)

        drop_results.append({
            'drop_ratio': drop_ratio,
            'features_kept': n_features_to_keep,
            'test_r2': test_r2,
            'test_mae': test_mae
        })

        print(f"{drop_ratio:9.1f} | {n_features_to_keep:12d} | {test_r2:7.3f} | {test_mae:8.3f}")

    # 找到最佳drop比例
    best_drop_result = max(drop_results, key=lambda x: x['test_r2'])
    print(f"\nBest drop ratio: {best_drop_result['drop_ratio']:.1f} (Test R²: {best_drop_result['test_r2']:.3f})")

    # 使用最佳drop比例重新训练最终模型
    best_n_features = best_drop_result['features_kept']
    best_features = feature_importance_df.head(best_n_features)['feature'].tolist()

    # 重新训练最终模型
    X_train_final = final_model._build_morgan_features(X_train_raw)[best_features]
    X_test_final = final_model._build_morgan_features(X_test_raw)[best_features]

    # Create final XGBoost model with best parameters
    final_xgb = XGBRegressor(
        learning_rate=best_params['learning_rate'],
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        subsample=best_params.get('subsample', 1.0),
        colsample_bytree=best_params.get('colsample_bytree', 1.0),
        reg_alpha=best_params.get('reg_alpha', 0),
        reg_lambda=best_params.get('reg_lambda', 1),
        gamma=best_params.get('gamma', 0),
        min_child_weight=best_params.get('min_child_weight', 1),
        random_state=42,
        n_jobs=-1
    )
    
    final_xgb.fit(X_train_final, y_train)
#    joblib.dump(final_xgb,'final_xgb.pkl')  # Changed filename
    y_pred_final = final_xgb.predict(X_test_final)

    print(f"\nFinal Model with Best Drop Ratio:")
    print(f"Final Test R² Score: {r2_score(y_test, y_pred_final):.3f}")
    print(f"Final Test MAE: {mean_absolute_error(y_test, y_pred_final):.3f} eV")
    print(f"Final Best Parameters: {best_params}")
    print(f"Best Drop Ratio: {best_drop_result['drop_ratio']:.1f}")
    print(f"Features Used: {len(best_features)}")

    # 保存结果
    submit_dir = os.environ.get("SLURM_SUBMIT_DIR", os.getcwd())

    # 保存超参数搜索结果
    results_df = pd.DataFrame({
        'round': range(1, len(best_score_history) + 1),
        'best_cv_score': best_score_history,
        'improvement': [0] + [best_score_history[i] - best_score_history[i-1]
                             for i in range(1, len(best_score_history))]
    })
    results_df.to_csv(os.path.join(submit_dir, "xgb_ecfp_optimization_results_t2.csv"), index=False)  # Changed filename

    # 保存drop比例测试结果
    drop_results_df = pd.DataFrame(drop_results)
    drop_results_df.to_csv(os.path.join(submit_dir, "xgb_drop_ratio_testing_t2.csv"), index=False)  # Changed filename

    # 保存特征重要性（基于最终使用的特征）
    final_feature_importances = final_xgb.feature_importances_  # Changed from final_gbr to final_xgb
    top_indices = np.argsort(final_feature_importances)[::-1][:20]
    top_features = [(best_features[i], final_feature_importances[i]) for i in top_indices]
    print("\nTop 20 important ECFP bits (after feature selection):")
    for fname, importance in top_features:
        print(f"Feature {fname}: {importance:.4f}")

    feat_imp_df = pd.DataFrame({
        "Feature": best_features,
        "Importance": final_feature_importances
    }).sort_values("Importance", ascending=False)
    feat_imp_df.to_csv(os.path.join(submit_dir, "xgb_ecfp_feature_importance_t2.csv"), index=False)  # Changed filename

    # 保存最终使用的特征列表
    final_features_df = pd.DataFrame({
        "Feature": best_features,
        "Original_Index": [int(f) for f in best_features]
    })
    final_features_df.to_csv(os.path.join(submit_dir, "xgb_final_selected_features_t2.csv"), index=False)  # Changed filename

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

    # Use best parameters to train XGBoost model for SHAP analysis
    print("   Training XGBoost model with best parameters for SHAP analysis...")

    best_params_for_shap = {
        'learning_rate': best_params['learning_rate'],
        'subsample': best_params.get('subsample', 1.0),
        'colsample_bytree': best_params.get('colsample_bytree', 1.0),
        'n_estimators': best_params['n_estimators'],
        'max_depth': best_params['max_depth'],
        'reg_alpha': best_params.get('reg_alpha', 0),
        'reg_lambda': best_params.get('reg_lambda', 1),
        'gamma': best_params.get('gamma', 0),
        'min_child_weight': best_params.get('min_child_weight', 1),
        'random_state': 42,
        'n_jobs': -1
    }
    # Create new XGBoost model
    xgb_for_shap = XGBRegressor(**best_params_for_shap)

    # Train model
    xgb_for_shap.fit(X_shap, y_train)

    print(f"   SHAP model trained with parameters: {best_params_for_shap}")

    # Use TreeExplainer for XGBoost
    print("   Creating SHAP explainer...")
    
    explainer = shap.TreeExplainer(xgb_for_shap)

    print("   SHAP explainer created successfully")
    
    # Calculate SHAP values for all features
    print(f"   Computing SHAP values for all {len(best_features)} features...")
    sample_size = min(100, len(X_test_shap))  # Use smaller sample for speed
    X_shap_sample = X_test_shap.iloc[:sample_size]
    shap_values = explainer.shap_values(X_shap_sample)
    
    # Calculate mean absolute SHAP values for each feature (SHAP importance)
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    
    # Create mapping from feature name to SHAP importance
    shap_importance_dict = {}
    for idx, feature_name in enumerate(best_features):
        shap_importance_dict[str(feature_name)] = mean_shap_values[idx]
    
    print(f"   Calculated SHAP importance for all {len(best_features)} features")
    print(f"   SHAP importance range: {mean_shap_values.min():.6f} to {mean_shap_values.max():.6f}")

    # STEP 2: Select top 20 features based on SHAP importance for visualization
    print("\n2. Selecting top 20 features based on SHAP importance for visualization...")
    
    # Sort features by SHAP importance and select top 20
    shap_importance_indices = np.argsort(mean_shap_values)[::-1][:20]
    top_20_features_by_shap = [best_features[i] for i in shap_importance_indices]
    top_20_shap_importances = [mean_shap_values[i] for i in shap_importance_indices]
    
    print(f"   Selected top 20 features based on SHAP importance from {len(best_features)} final selected features")
    print(f"   Top 5 features by SHAP: {top_20_features_by_shap[:5]} with SHAP importances: {[f'{imp:.4f}' for imp in top_20_shap_importances[:5]]}")

    # STEP 3: Analyze molecular fragments for selected features
    print("\n3. Analyzing Morgan fingerprint bit patterns for top 20 features (by SHAP importance)...")

    # Get Morgan fingerprint features from training data
    X_morgan_train = ecfp_model._build_morgan_features(X_train_raw)

    # Analyze activation patterns for selected features - using SHAP importance
    morgan_analysis = {}

    for feature_name in top_20_features_by_shap:
        # Get SHAP importance for this feature
        shap_importance = shap_importance_dict[str(feature_name)]
        # 确保特征名称是字符串类型
        feature_name_str = str(feature_name)
        try:
            feature_idx = int(feature_name)
            bit_values = X_morgan_train.iloc[:, feature_idx]
        except (ValueError, IndexError):
            # 如果无法转换为索引，尝试通过列名查找
            if feature_name_str in X_morgan_train.columns:
                bit_values = X_morgan_train[feature_name_str]
            else:
                print(f"Warning: Could not find feature {feature_name_str}, skipping...")
                continue

        # 统计激活该位的分子数量
        activated_mols = bit_values[bit_values == 1]
        deactivated_mols = bit_values[bit_values == 0]

        # Find molecule indices that activate this bit
        activated_indices = bit_values[bit_values == 1].index.tolist()

        morgan_analysis[feature_name_str] = {
            'importance': shap_importance,  # Use SHAP importance instead of feature importance
            'shap_importance': shap_importance,  # Store SHAP importance explicitly
            'activation_rate': len(activated_mols) / len(bit_values),
            'activated_count': len(activated_mols),
            'activated_indices': activated_indices,
            'bit_values': bit_values.values
        }

    print(f"Analyzed {len(morgan_analysis)} top 20 features (by SHAP importance)")

    # STEP 4: Create Morgan fingerprint importance visualization using SHAP importance
    print("\n4. Creating Morgan fingerprint importance visualization (using SHAP importance)...")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Subplot 1: Morgan fingerprint importance bar chart (using SHAP importance)
    feature_names = [f"Bit_{f}" for f in top_20_features_by_shap]
    importances = [morgan_analysis[str(f)]['importance'] for f in top_20_features_by_shap]

    bars = ax1.barh(range(len(feature_names)), importances, color='lightblue', alpha=0.8)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('SHAP Feature Importance')
    ax1.set_title(f'Top 20 Morgan Fingerprint Bits - SHAP Importance ({len(top_20_features_by_shap)} features)')
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels on bar chart
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{imp:.4f}', ha='left', va='center', fontsize=8)
    # Subplot 2: Morgan fingerprint activation rate bar chart
    activation_rates = [morgan_analysis[str(f)]['activation_rate'] for f in top_20_features_by_shap]
    activation_bars = ax2.barh(range(len(feature_names)), activation_rates,
                              color='lightcoral', alpha=0.8)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(feature_names)
    ax2.set_xlabel('Activation Rate')
    ax2.set_title(f'Morgan Fingerprint Bit Activation Rates - Top 20 Features by SHAP Importance ({len(top_20_features_by_shap)} features)')
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels on bar chart
    for i, (bar, rate) in enumerate(zip(activation_bars, activation_rates)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{rate:.3f}', ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(submit_dir, 'xgb_ecfp_importance_visualization_shap.png'),
                dpi=600, bbox_inches='tight')
    plt.show()

    # STEP 5: Analyze molecular structure information for selected features
    print("\n5. Analyzing Morgan fingerprint molecular substructures for top 20 features (by SHAP importance)...")

    # Get molecular samples that activate each Morgan fingerprint bit
    morgan_substructures = {}

    for feature_name in top_20_features_by_shap:  # Use top 20 by SHAP importance
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

        # Find molecule indices that activate this bit
        activated_indices = bit_values[bit_values == 1].index.tolist()

        if len(activated_indices) > 0:
            # Select first 3 molecules that activate this bit for analysis
            n_molecules_to_analyze = min(3, len(activated_indices))
            selected_indices = activated_indices[:n_molecules_to_analyze]

            print(f"Feature {feature_name}: Analyzing {n_molecules_to_analyze} molecules out of {len(activated_indices)} activated molecules")

            for mol_idx, sample_idx in enumerate(selected_indices):
                sample_smiles = X_train_raw.iloc[sample_idx]['smiles']
                sample_mol = Chem.MolFromSmiles(sample_smiles)
                if sample_mol is not None:
                    # Get atomic environment information for Morgan fingerprint
                    try:
                        radius = int(ecfp_model.radius)
                        print(f"  Analyzing molecule {mol_idx+1}/{n_molecules_to_analyze} for bit {feature_name} with radius {radius}")
                        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=2**ecfp_model.n_bits)
                        fp = morgan_gen.GetFingerprint(sample_mol)

                        # Get AllChem bit information as reference
                        allchem_bit_info = {}
                        detailed_fp = AllChem.GetMorganFingerprint(sample_mol, radius, bitInfo=allchem_bit_info)

                        # Create unique key for each molecule
                        mol_key = f"{feature_name}_mol_{mol_idx+1}"
                        morgan_substructures[mol_key] = {
                            'feature_name': feature_name,
                            'molecule_index': mol_idx + 1,
                            'sample_smiles': sample_smiles,
                            'sample_mol': sample_mol,
                            'activation_rate': len(activated_mols) / len(bit_values),
                            'bit_info': allchem_bit_info,
                            'morgan_fp': fp,
                            'detailed_fp': detailed_fp,
                            'allchem_bit_info': allchem_bit_info
                        }

                    except Exception as e:
                        print(f"  Warning: Could not analyze molecule {mol_idx+1} for bit {feature_name}: {e}")
                        mol_key = f"{feature_name}_mol_{mol_idx+1}"
                        morgan_substructures[mol_key] = {
                            'feature_name': feature_name,
                            'molecule_index': mol_idx + 1,
                            'sample_smiles': sample_smiles,
                            'sample_mol': sample_mol,
                            'activation_rate': len(activated_mols) / len(bit_values),
                            'bit_info': {},
                            'morgan_fp': None,
                            'detailed_fp': None
                        }

    print(f"Analyzed molecular substructures for {len(morgan_substructures)} features")
    # STEP 6: Visualize molecular substructures corresponding to Morgan fingerprint bits
    print("\n6. Creating molecular substructure visualizations for Morgan fingerprint bits...")

    # Use top 20 features selected by SHAP importance
    top_features_to_show = top_20_features_by_shap

    # 按特征分组处理分子
    for feature_name in top_features_to_show:
        print(f"\n=== Processing feature {feature_name} ===")

        # Find all molecules for this feature
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
                print(f"    Available bits in bit_info: {list(bit_info.keys())[:30]}...")  # Show first 30 bits

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

    # STEP 7: Create statistical information visualization for Morgan fingerprint bits
    print("\n7. Creating Morgan fingerprint statistics visualization (using SHAP importance)...")

    # Count activation patterns for each bit (using top 20 by SHAP)
    bit_stats = []
    for feature_name in top_20_features_by_shap:  # Use top 20 by SHAP importance
        if str(feature_name) in morgan_analysis:
            stats = morgan_analysis[str(feature_name)]
            bit_stats.append({
                'Bit': feature_name,
                'Importance': stats['importance'],
                'Activation_Rate': stats['activation_rate'],
                'Activated_Count': stats['activated_count']
            })
    if bit_stats:
        # 创建统计图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Subplot 1: SHAP Importance vs activation rate scatter plot
        importances = [s['Importance'] for s in bit_stats]
        activation_rates = [s['Activation_Rate'] for s in bit_stats]
        ax1.scatter(activation_rates, importances, alpha=0.7, s=50)
        ax1.set_xlabel('Activation Rate')
        ax1.set_ylabel('SHAP Feature Importance')
        ax1.set_title('SHAP Feature Importance vs Activation Rate')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Activation rate distribution histogram
        ax2.hist(activation_rates, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Activation Rate')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Activation Rates')
        ax2.grid(True, alpha=0.3)

        # Subplot 3: SHAP Importance distribution histogram
        ax3.hist(importances, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_xlabel('SHAP Feature Importance')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of SHAP Feature Importance')
        ax3.grid(True, alpha=0.3)

        # Subplot 4: Activation count vs SHAP importance
        activated_counts = [s['Activated_Count'] for s in bit_stats]
        ax4.scatter(activated_counts, importances, alpha=0.7, s=50, color='green')
        ax4.set_xlabel('Number of Activated Molecules')
        ax4.set_ylabel('SHAP Feature Importance')
        ax4.set_title('SHAP Feature Importance vs Activation Count')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(submit_dir, 'morgan_fingerprint_statistics.png'),
                   dpi=600, bbox_inches='tight')
        plt.show()

        print("Created Morgan fingerprint statistics visualization")
    
    # STEP 8: Create SHAP summary plot for top 10 features
    print("\n8. Creating SHAP summary plot for top 10 features...")

    # 1. Grab strictly the first 10 indices and names from your original list
    top_10_indices = shap_importance_indices[:10]
    top_10_features = top_20_features_by_shap[:10]

    # 2. Subset the arrays so SHAP ONLY sees these 10 features
    shap_values_top10 = shap_values[:, top_10_indices]
    X_shap_sample_top10 = X_shap_sample.iloc[:, top_10_indices]

    plt.figure(figsize=(12, 6)) # slightly shorter figure for 10 features
    
    # 3. Plot the subsetted data 
    # (Notice we pass the _top10 variables, NOT the full shap_values)
    shap.summary_plot(
        shap_values_top10, 
        X_shap_sample_top10,
        feature_names=[f"Bit_{f}" for f in top_10_features],
        show=False
    )
                      
    plt.title('SHAP Summary Plot - Top 10 Morgan Fingerprint Bits')
    plt.tight_layout()
    
    plt.savefig(os.path.join(submit_dir, 'shap_summary_plot_top10.png'), dpi=600, bbox_inches='tight')
    plt.show()

    # STEP 9: Create SHAP dependency plots (select top 5 most important features)
    print("\n9. Creating SHAP dependency plots...")

    # Select top 5 most important features for dependency plot analysis
    top_5_features = top_20_features_by_shap[:5]
    top_5_shap_importances = top_20_shap_importances[:5]

    for i, (feature_name, feature_idx) in enumerate(zip(top_5_features, range(5))):
        print(f"Creating dependency plot for feature {feature_name} (SHAP importance: {top_5_shap_importances[i]:.4f})...")

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, shap_values_top20, X_shap_sample_top20,
                           feature_names=[f"Bit_{f}" for f in top_20_features_by_shap],
                           show=False)
        plt.title(f'SHAP Dependence Plot - Morgan Fingerprint Bit {feature_name}\nSHAP Importance: {top_5_shap_importances[i]:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(submit_dir, f'shap_dependence_{feature_name}.png'),
                   dpi=600, bbox_inches='tight')
        plt.show()

    # STEP 10: Save SHAP analysis results
    print("\n10. Saving SHAP analysis results...")

    # Create SHAP importance DataFrame - based on top 20 features sorted by SHAP importance
    shap_importance_df = pd.DataFrame({
        'Feature': [f"Bit_{f}" for f in top_20_features_by_shap],
        'Original_Index': top_20_features_by_shap,
        'SHAP_Importance': top_20_shap_importances,
        'Feature_Rank': range(1, len(top_20_features_by_shap) + 1)
    }).sort_values('SHAP_Importance', ascending=False)

    shap_importance_df.to_csv(os.path.join(submit_dir, 'shap_importance_analysis_top20.csv'), index=False)

    print("SHAP analysis completed successfully!")
    print(f"Analyzed top 20 features (by SHAP importance) from {len(best_features)} final selected features")

    # STEP 11: Save Morgan fingerprint analysis results
    print("\n11. Saving Morgan fingerprint analysis results for top 20 features (by SHAP importance)...")

    morgan_analysis_df = pd.DataFrame([
        {
            'Morgan_Bit': f,
            'SHAP_Importance': morgan_analysis[str(f)]['importance'],  # This is SHAP importance now
            'Activation_Rate': morgan_analysis[str(f)]['activation_rate'],
            'Activated_Count': morgan_analysis[str(f)]['activated_count'],
            'Total_Molecules': len(morgan_analysis[str(f)]['bit_values']),
            'Sample_SMILES': morgan_substructures.get(f"{f}_mol_1", {}).get('sample_smiles', 'N/A')
        }
        for f in top_20_features_by_shap  # Use top 20 by SHAP importance
    ])

    morgan_analysis_df.to_csv(os.path.join(submit_dir, 'morgan_detailed_analysis_shap_top20.csv'), index=False)

    # STEP 12: Save molecular substructure information
    print("\n12. Saving molecular substructure information for top 20 features (by SHAP importance)...")

    substructure_info = []
    for feature_name in top_20_features_by_shap:  # Use top 20 by SHAP importance
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
    substructure_df.to_csv(os.path.join(submit_dir, 'morgan_substructure_analysis_shap_top20.csv'), index=False)
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


