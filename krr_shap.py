import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem import rdFingerprintGenerator
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.inspection import permutation_importance
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

class ECFPKRR(BaseEstimator, RegressorMixin):
    """Kernel Ridge Regressor with Morgan fingerprints."""

    def __init__(self, radius=3, n_bits=12, alpha=0.9):
        self.radius = radius
        self.n_bits = n_bits  # n_bits=12 means fpSize=4096 (2^12)
        self.alpha = alpha
        self.krr_model = None
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
        self.krr_model = KernelRidge(
            alpha=self.alpha
        )

        # Train the model
        self.krr_model.fit(X_morgan, y)
        return self

    def predict(self, X):
        """预测"""
        if self.krr_model is None:
            raise ValueError("Model not fitted yet")

        X_morgan = self._build_morgan_features(X)
        # 确保特征顺序一致
        if self.feature_names_ is not None:
            X_morgan = X_morgan[self.feature_names_]

        return self.krr_model.predict(X_morgan)

    def get_feature_importances(self):
        """Kernel Ridge 不提供树模型的特征重要性，返回 None"""
        return None


def refine(param, val):
    if isinstance(val, int):
        # 这里只对 KRR 相关的整数参数（如 radius, n_bits）做细化
        if param in ['radius', 'n_bits']:
            return sorted(set([max(1, val-2), max(1, val-1), val, val+1, val+2]))
        else:
            return [val]

    if isinstance(val, float):
        # 对 alpha 等正浮点超参数做对数尺度附近的细化
        vals = [max(1e-10, val/2), max(1e-10, val*0.8), val, val*1.2, val*2]
        return sorted(set([round(v, 6) for v in vals if v > 0.0]))

    if isinstance(val, str):
        return [val]
    if val is None:
        return [None]
    return [val]

param_distributions = {
    'radius': [2, 3, 4],
    'n_bits': [9, 10, 11, 12],  # Corresponds to 2^9=512 up to 2^12=4096
    'alpha': [1e-3, 1e-2, 1e-1, 1.0, 10.0]  # Regularization strength for Kernel Ridge
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

    # 创建并训练Morgan指纹 KRR 模型
    krr = ECFPKRR()

    # 使用随机搜索替代网格搜索
    RS = RandomizedSearchCV(
        estimator=krr,
        param_distributions=current_param_distributions,
        n_iter=100,
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
        new_distributions[k] = refine(k, best[k])
    param_distributions = new_distributions

    print(f"Refined parameter distributions for next round")

# 最终评估
if best_model is not None:
    # 使用最佳参数重新训练最终模型（KRR）
    final_model = ECFPKRR(**best_params)
    final_model.fit(X_train_raw, y_train)

    # 用最终 KRR 模型计算特征重要性（permutation importance）
    X_train_morgan_full = final_model._build_morgan_features(X_train_raw)

    result = permutation_importance(
        final_model.krr_model,       # 内部的 KRR 模型
        X_train_morgan_full,
        y_train,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    importances = result.importances_mean
    final_feature_importances = importances

    feature_importance_df = pd.DataFrame({
        'feature': X_train_morgan_full.columns.astype(str),
        'importance': importances
    }).sort_values('importance', ascending=False)

    # 保存最终模型
    joblib.dump(final_model, 'final_krr.pkl')

    print(f"\nBest parameters from randomized search: {best_params}")

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

    krr_selected = KernelRidge(
        alpha=best_params['alpha']
    )

    krr_selected.fit(X_train_selected, y_train)
    y_pred = krr_selected.predict(X_test_selected)

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

    final_krr = KernelRidge(
        alpha=best_params['alpha']
    )
    final_krr.fit(X_train_final, y_train)
#    joblib.dump(final_krr,'final_krr.pkl')
    y_pred_final = final_krr.predict(X_test_final)

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
    results_df.to_csv(os.path.join(submit_dir, "krr_ecfp_optimization_results_t2.csv"), index=False)

    # 保存drop比例测试结果
    drop_results_df = pd.DataFrame(drop_results)
    drop_results_df.to_csv(os.path.join(submit_dir, "krr_drop_ratio_testing_t2.csv"), index=False)

    # KRR 没有树模型特征重要性，这里只保存最终使用的特征列表
    final_features_df = pd.DataFrame({
        "Feature": best_features,
        "Original_Index": [int(f) for f in best_features]
    })
    final_features_df.to_csv(os.path.join(submit_dir, "krr_final_selected_features_t2.csv"), index=False)

else:
    print("No valid model was trained!")

# 定义可视化函数（全局作用域）
def visualize_ecfp_importance(ecfp_model, best_features, feature_importances,
                             X_train_raw, X_test_raw, y_train, submit_dir, best_params):
    """可视化Morgan指纹重要性和分子结构"""

    print("\n=== Morgan Fingerprint Visualization and Molecular Structure Analysis ===")

    # 1. 分析Morgan指纹位对应的分子片段
    print("\n1. Analyzing Morgan fingerprint bit patterns for final selected features...")

    # 获取训练数据的Morgan指纹特征
    X_morgan_train = ecfp_model._build_morgan_features(X_train_raw)

    # 分析每个重要Morgan指纹位的激活模式 - 使用最终选择的特征
    morgan_analysis = {}

    for i, (feature_name, importance) in enumerate(zip(best_features, feature_importances)):
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

        # 找到激活该位的分子索引
        activated_indices = bit_values[bit_values == 1].index.tolist()

        morgan_analysis[feature_name_str] = {
            'importance': importance,
            'activation_rate': len(activated_mols) / len(bit_values),
            'activated_count': len(activated_mols),
            'activated_indices': activated_indices,
            'bit_values': bit_values.values
        }

    print(f"Analyzed {len(morgan_analysis)} final selected Morgan fingerprint features")

    # 2. 创建Morgan指纹重要性可视化
    print("\n2. Creating Morgan fingerprint importance visualization for final selected features...")

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # 子图1: Morgan指纹重要性条形图
    feature_names = [f"Bit_{f}" for f in best_features]
    importances = [morgan_analysis[str(f)]['importance'] for f in best_features]

    bars = ax1.barh(range(len(feature_names)), importances, color='lightblue', alpha=0.8)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels(feature_names)
    ax1.set_xlabel('Feature Importance')
    ax1.set_title(f'Final Selected Morgan Fingerprint Bits - Feature Importance ({len(best_features)} features)')
    ax1.grid(axis='x', alpha=0.3)

    # 在条形图上添加数值标签
    for i, (bar, imp) in enumerate(zip(bars, importances)):
        ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{imp:.4f}', ha='left', va='center', fontsize=8)

    # 子图2: Morgan指纹激活率条形图
    activation_rates = [morgan_analysis[str(f)]['activation_rate'] for f in best_features]
    activation_bars = ax2.barh(range(len(feature_names)), activation_rates,
                               color='lightcoral', alpha=0.8)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(feature_names)
    ax2.set_xlabel('Activation Rate')
    ax2.set_title(f'Morgan Fingerprint Bit Activation Rates - Final Selected Features ({len(best_features)} features)')
    ax2.grid(axis='x', alpha=0.3)

    # 在条形图上添加数值标签
    for i, (bar, rate) in enumerate(zip(activation_bars, activation_rates)):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{rate:.3f}', ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(submit_dir, 'ecfp_importance_visualization.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 分析Morgan指纹位的分子结构信息
    print("\n3. Analyzing Morgan fingerprint molecular substructures for final selected features...")

    # 获取激活每个Morgan指纹位的分子样本
    morgan_substructures = {}

    for feature_name in best_features[:10]:  # 限制为前10个特征
        activated_indices = morgan_analysis[str(feature_name)]['activated_indices']

        if len(activated_indices) > 0:
            # 选择前3个激活该位的分子进行分析（而不是只选择第一个）
            n_molecules_to_analyze = min(3, len(activated_indices))
            selected_indices = activated_indices[:n_molecules_to_analyze]

            print(f"Feature {feature_name}: Analyzing {n_molecules_to_analyze} molecules out of {len(activated_indices)} activated molecules")

            for mol_idx, sample_idx in enumerate(selected_indices):
                sample_smiles = X_train_raw.iloc[sample_idx]['smiles']
                sample_mol = Chem.MolFromSmiles(sample_smiles)
                if sample_mol is not None:
                    # 获取Morgan指纹的原子环境信息 - 使用正确的参数
                    try:
                        # 确保radius是有效的整数，使用训练时的radius参数
                        radius = int(ecfp_model.radius)
                        print(f"  Analyzing molecule {mol_idx+1}/{n_molecules_to_analyze} for bit {feature_name} with radius {radius}")

                        # 使用与训练时相同的Morgan指纹生成方法
                        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
                            radius=radius,
                            fpSize=2**ecfp_model.n_bits
                        )
                        fp = morgan_gen.GetFingerprint(sample_mol)

                        # 这里我们使用AllChem作为备选方案，但需要建立正确的映射关系

                        # 获取AllChem的位信息作为参考
                        allchem_bit_info = {}
                        detailed_fp = AllChem.GetMorganFingerprint(sample_mol, radius, bitInfo=allchem_bit_info)

                        # 直接使用AllChem的位信息，不再尝试手动构建
                        # 因为AllChem.GetMorganFingerprint已经提供了完整的位信息
                        # 为每个分子创建唯一的键
                        mol_key = f"{feature_name}_mol_{mol_idx+1}"
                        morgan_substructures[mol_key] = {
                            'feature_name': feature_name,
                            'molecule_index': mol_idx + 1,
                            'sample_smiles': sample_smiles,
                            'sample_mol': sample_mol,
                            'activation_rate': morgan_analysis[str(feature_name)]['activation_rate'],
                            'bit_info': allchem_bit_info,  # 直接使用AllChem的位信息
                            'morgan_fp': fp,
                            'detailed_fp': detailed_fp,
                            'allchem_bit_info': allchem_bit_info  # 保存AllChem的位信息作为参考
                        }

                    except Exception as e:
                        print(f"  Warning: Could not analyze molecule {mol_idx+1} for bit {feature_name}: {e}")
                        # 如果分析失败，仍然保存基本信息
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

    # 4. 可视化Morgan指纹对应的分子子结构
    print("\n4. Creating molecular substructure visualizations for Morgan fingerprint bits...")

    # 为前10个最重要的特征创建分子子结构图
    n_features_to_show = min(10, len(best_features))
    top_features_to_show = best_features[:n_features_to_show]

    # 按特征分组处理分子
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

                # 为分子添加原子序号，便于观察
                mol_copy = Chem.Mol(mol)
                for atom in mol_copy.GetAtoms():
                    atom.SetAtomMapNum(atom.GetIdx())

                # 使用AllChem生成Morgan指纹并获取位信息
                radius = int(ecfp_model.radius)
                n_bits = 2**ecfp_model.n_bits

                # 创建位信息字典
                bit_info = {}

                # 生成Morgan指纹并填充位信息
                fp = AllChem.GetMorganFingerprintAsBitVect(mol_copy, radius, nBits=n_bits, bitInfo=bit_info)

                print(f"    Generated Morgan fingerprint with radius={radius}, n_bits={n_bits}")
                print(f"    Fingerprint length: {fp.GetNumBits()}")
                print(f"    Activated bits: {fp.GetNumOnBits()}")

                # 检查位信息字典的内容
                if not bit_info:
                    print(f"    Warning: No bit information generated for molecule {mol_idx}")
                    continue

                # 尝试找到与当前特征对应的位
                target_bit = None
                contributing_substructures = []
                print(f"    Available bits in bit_info: {list(bit_info.keys())[:10]}...")  # 显示前10个位

                # 方法1: 直接使用特征名称作为位索引
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

                # 方法2: 如果直接匹配失败，尝试找到最相似的位
                if target_bit is None and bit_info:
                    # 选择第一个可用的位进行分析
                    target_bit = list(bit_info.keys())[0]
                    contributing_substructures = bit_info[target_bit]
                    print(f"    Using fallback bit {target_bit} with {len(contributing_substructures)} contributing substructures")

                if target_bit is not None and contributing_substructures:
                    print(f"    Analyzing Bit {target_bit}")
                    print(f"    Contributing substructures: {len(contributing_substructures)}")

                    # 分析每个子结构的中心原子类型和每层半径的原子类型
                    substructure_analysis = []

                    for i, (atom_idx, radius_val) in enumerate(contributing_substructures):
                        print(f"      - 子结构 {i+1}: 中心原子索引 = {atom_idx}, 半径 = {radius_val}")

                        # 获取中心原子信息
                        center_atom = mol_copy.GetAtomWithIdx(atom_idx)
                        center_atom_symbol = center_atom.GetSymbol()
                        center_atom_degree = center_atom.GetDegree()
                        center_atom_valence = center_atom.GetTotalValence()

                        print(f"        中心原子: {center_atom_symbol} (索引{atom_idx}), 度={center_atom_degree}, 价={center_atom_valence}")

                        # 分析每层半径的原子类型
                        layer_analysis = {}
                        for r in range(radius_val + 1):
                            layer_atoms = set()
                            layer_bonds = set()

                            if r == 0:
                                # 中心原子
                                layer_atoms.add(atom_idx)
                            else:
                                # 使用FindAtomEnvironmentOfRadiusN找到指定半径的环境
                                env = Chem.FindAtomEnvironmentOfRadiusN(mol_copy, r, atom_idx)
                                if env is not None:
                                    for bond_idx in env:
                                        bond = mol_copy.GetBondWithIdx(bond_idx)
                                        layer_bonds.add(bond_idx)
                                        layer_atoms.add(bond.GetBeginAtomIdx())
                                        layer_atoms.add(bond.GetEndAtomIdx())
                            # 统计该层的原子类型
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

                            print(f"        半径{r}: 原子类型={atom_types}, 原子数={len(layer_atoms)}, 键数={len(layer_bonds)}")

                        # 保存子结构分析结果
                        substructure_analysis.append({
                            'center_atom_idx': atom_idx,
                            'center_atom_symbol': center_atom_symbol,
                            'center_atom_degree': center_atom_degree,
                            'center_atom_valence': center_atom_valence,
                            'radius': radius_val,
                            'layer_analysis': layer_analysis
                        })

                    # 创建指纹示意图
                    print(f"    Creating fingerprint schematic for bit {target_bit}...")

                    try:
                        # 创建指纹示意图
                        fig, axes = plt.subplots(1, len(substructure_analysis), figsize=(4*len(substructure_analysis), 6))
                        if len(substructure_analysis) == 1:
                            axes = [axes]

                        for i, substructure in enumerate(substructure_analysis):
                            ax = axes[i]

                            # 绘制中心原子
                            center_symbol = substructure['center_atom_symbol']
                            center_degree = substructure['center_atom_degree']

                            # 创建圆形图显示中心原子信息
                            ax.text(0.5, 0.8, f"中心原子: {center_symbol}", ha='center', va='center', fontsize=12, weight='bold')
                            ax.text(0.5, 0.6, f"度: {center_degree}", ha='center', va='center', fontsize=10)
                            ax.text(0.5, 0.4, f"价: {substructure['center_atom_valence']}", ha='center', va='center', fontsize=10)
                            # 绘制各层原子类型分布
                            y_pos = 0.2
                            for r in range(substructure['radius'] + 1):
                                if r in substructure['layer_analysis']:
                                    layer_info = substructure['layer_analysis'][r]
                                    atom_types = layer_info['atom_types']
                                    atom_count = sum(atom_types.values())

                                    ax.text(0.5, y_pos, f"半径{r}: {atom_types}", ha='center', va='center', fontsize=9)
                                    y_pos -= 0.15

                            ax.set_xlim(0, 1)
                            ax.set_ylim(0, 1)
                            ax.set_title(f'子结构 {i+1}', fontsize=12)
                            ax.axis('off')

                        plt.suptitle(f'Morgan指纹位 {target_bit} 示意图\n分子{mol_idx}, 半径={radius}, 子结构数={len(substructure_analysis)}', fontsize=14)
                        plt.tight_layout()

                        # 保存指纹示意图
                        schematic_filename = os.path.join(submit_dir, f'morgan_bit_{feature_name}_mol{mol_idx}_schematic.png')
                        plt.savefig(schematic_filename, dpi=300, bbox_inches='tight')
                        print(f"    Saved fingerprint schematic to {schematic_filename}")
                        plt.close()

                    except Exception as e:
                        print(f"    Error creating fingerprint schematic: {e}")

                    # 为每个贡献的子结构找到构成该子结构的所有原子和键
                    highlight_list = []

                    for i, (atom_idx, radius_val) in enumerate(contributing_substructures):
                        print(f"      - 子结构 {i+1}: 中心原子索引 = {atom_idx}, 半径 = {radius_val}")

                        # 找到构成该子结构的所有原子和键
                        env = Chem.FindAtomEnvironmentOfRadiusN(mol_copy, radius_val, atom_idx)
                        atoms_in_env = set()
                        bonds_to_highlight = set()

                        if env is not None:
                            for bond_idx in env:
                                bonds_to_highlight.add(bond_idx)
                                bond = mol_copy.GetBondWithIdx(bond_idx)
                                atoms_in_env.add(bond.GetBeginAtomIdx())
                                atoms_in_env.add(bond.GetEndAtomIdx())
                            # 将需要高亮的原子和键的信息保存起来
                            highlight_list.append((list(atoms_in_env), list(bonds_to_highlight)))
                            print(f"        环境中的原子: {list(atoms_in_env)}")
                            print(f"        环境中的键: {list(bonds_to_highlight)}")
                        else:
                            print(f"        无法找到半径 {radius_val} 的原子环境")

                    # 可视化贡献了target_bit的子结构
                    if highlight_list:
                        print(f"    Creating visualization with {len(highlight_list)} substructures...")

                        # 创建网格图像显示多个子结构
                        try:
                            # 使用Draw.MolsToGridImage创建网格图像
                            grid_image = Draw.MolsToGridImage(
                                [mol_copy for _ in highlight_list],  # 显示原始分子用于绘图
                                highlightAtomLists=[h[0] for h in highlight_list],
                                highlightBondLists=[h[1] for h in highlight_list],
                                legends=[f"子结构 {i+1} (位 {target_bit})" for i in range(len(highlight_list))],
                                molsPerRow=min(3, len(highlight_list)),
                                subImgSize=(300, 300)
                            )

                            # 保存网格图像
                            grid_filename = os.path.join(submit_dir, f'morgan_bit_{feature_name}_mol{mol_idx}_substructures.png')
                            grid_image.save(grid_filename)
                            print(f"    Saved substructure grid visualization to {grid_filename}")

                        except Exception as e:
                            print(f"    Error creating grid image: {e}")
                            # 如果网格图像创建失败，尝试创建单个图像
                            try:
                                # 创建单个子结构的可视化
                                for i, (atoms, bonds) in enumerate(highlight_list):
                                    img = Draw.MolToImage(
                                        mol_copy,
                                        highlightAtoms=atoms,
                                        highlightBonds=bonds,
                                        highlightAtomColors={idx: (0.3, 0.7, 1.0) for idx in atoms},
                                        highlightBondColors={idx: (0.3, 0.7, 1.0) for idx in bonds}
                                    )

                                    single_filename = os.path.join(
                                        submit_dir,
                                        f'morgan_bit_{feature_name}_mol{mol_idx}_substructure_{i+1}.png'
                                    )
                                    img.save(single_filename)
                                    print(f"    Saved single substructure visualization to {single_filename}")

                            except Exception as e2:
                                print(f"    Error creating single image: {e2}")
                    else:
                        print(f"    No valid substructures found for visualization")
                else:
                    print(f"    No valid bit information found for molecule {mol_idx}")

    print(f"Created molecular structure visualizations for {n_features_to_show} features")

    # 4.1. 创建Morgan指纹位的统计信息可视化
    print("\n4.1. Creating Morgan fingerprint statistics visualization...")

    # 统计每个位的激活模式
    bit_stats = []
    for feature_name in best_features[:10]:  # 限制为前10个特征
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

        # 子图1: 重要性 vs 激活率散点图
        importances = [s['Importance'] for s in bit_stats]
        activation_rates = [s['Activation_Rate'] for s in bit_stats]
        ax1.scatter(activation_rates, importances, alpha=0.7, s=50)
        ax1.set_xlabel('Activation Rate')
        ax1.set_ylabel('Feature Importance')
        ax1.set_title('Feature Importance vs Activation Rate')
        ax1.grid(True, alpha=0.3)

        # 子图2: 激活率分布直方图
        ax2.hist(activation_rates, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Activation Rate')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Activation Rates')
        ax2.grid(True, alpha=0.3)

        # 子图3: 重要性分布直方图
        ax3.hist(importances, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        ax3.set_xlabel('Feature Importance')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Feature Importance')
        ax3.grid(True, alpha=0.3)
        # 子图4: 激活数量 vs 重要性
        activated_counts = [s['Activated_Count'] for s in bit_stats]
        ax4.scatter(activated_counts, importances, alpha=0.7, s=50, color='green')
        ax4.set_xlabel('Number of Activated Molecules')
        ax4.set_ylabel('Feature Importance')
        ax4.set_title('Feature Importance vs Activation Count')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(submit_dir, 'morgan_fingerprint_statistics.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()

        print("Created Morgan fingerprint statistics visualization")
    # 5. SHAP analysis (directly on all features, then select top 10 by SHAP importance)
    print("\n5. Performing SHAP analysis on all features from final model...")

    # Prepare SHAP analysis data - use all final selected features
    print("Preparing SHAP analysis on all final selected features...")

    # Build Morgan fingerprint features using all final selected features
    X_shap_morgan = ecfp_model._build_morgan_features(X_train_raw)
    X_shap = X_shap_morgan[best_features]  # Use all final selected features

    # Build test features
    X_test_shap_morgan = ecfp_model._build_morgan_features(X_test_raw)
    X_test_shap = X_test_shap_morgan[best_features]

    # Use best parameters to train KRR model for SHAP analysis
    print("Training KRR model with best parameters for SHAP analysis...")

    best_params_for_shap = {
        'alpha': best_params['alpha'],
    }  

    # Create new KRR model
    krr_for_shap = KernelRidge(**best_params_for_shap)

    # Train model
    krr_for_shap.fit(X_shap, y_train)

    print(f"SHAP model trained with parameters: {best_params_for_shap}")

    # 创建SHAP解释器
    print("Creating SHAP explainer...")

    # 计算SHAP值（使用较小的样本以提高速度）
    sample_size = min(100, len(X_shap))
    X_shap_sample = X_shap.sample(n=sample_size, random_state=42)

    # 对于 KRR，使用 KernelExplainer 而不是 TreeExplainer
    explainer = shap.KernelExplainer(krr_for_shap.predict, X_shap_sample)
    # Calculate SHAP values for all features
    print(f"Computing SHAP values for all {len(best_features)} features...")
    sample_size = min(100, len(X_test_shap))  # Use smaller sample for speed
    X_shap_sample = X_test_shap.iloc[:sample_size]

    # For KRR model, use KernelExplainer with appropriate parameters
    explainer = shap.KernelExplainer(krr_for_shap.predict, X_shap_sample)
    shap_values = explainer.shap_values(X_shap_sample, nsamples=100)

    # Calculate mean absolute SHAP values for each feature
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Sort features by SHAP importance and select top 10
    shap_importance_indices = np.argsort(mean_shap_values)[::-1][:10]
    top_10_features = [best_features[i] for i in shap_importance_indices]
    top_10_shap_importances = [mean_shap_values[i] for i in shap_importance_indices]

    print(f"Selected top 10 features based on SHAP importance from {len(best_features)} final selected features")
    print(f"Top 5 features by SHAP: {top_10_features[:5]} with SHAP importances: {[f'{imp:.4f}' for imp in top_10_shap_importances[:5]]}")

    # Extract SHAP values for top 10 features
    shap_values_top10 = shap_values[:, shap_importance_indices]
    X_shap_sample_top10 = X_shap_sample.iloc[:, shap_importance_indices]

    # Create SHAP summary plot
    print("\n6. Creating SHAP summary plot for top 10 features...")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_top10, X_shap_sample_top10,
                      feature_names=[f"Bit_{f}" for f in top_10_features],
                      show=False)
    plt.title(f'SHAP Summary Plot - Top 10 Morgan Fingerprint Bits by SHAP Importance ({len(top_10_features)} features)')
    plt.tight_layout()
    plt.savefig(os.path.join(submit_dir, 'shap_summary_plot_top10.png'),
                dpi=600, bbox_inches='tight')
    plt.show()

    # Create SHAP dependency plots (select top 5 most important features)
    print("\n7. Creating SHAP dependency plots...")

    # Select top 10 most important features for dependency plot analysis
    top_10_features = top_10_features[:10]
    top_10_shap_importances = top_10_shap_importances[:10]

    for i, (feature_name, feature_idx) in enumerate(zip(top_10_features, range(10))):
        print(f"Creating dependency plot for feature {feature_name} (SHAP importance: {top_10_shap_importances[i]:.4f})...")

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature_idx, shap_values_top10, X_shap_sample_top10,
                           feature_names=[f"Bit_{f}" for f in top_10_features],
                           show=False)
        plt.title(f'SHAP Dependence Plot - Morgan Fingerprint Bit {feature_name}\nSHAP Importance: {top_10_shap_importances[i]:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(submit_dir, f'shap_dependence_{feature_name}.png'),
                   dpi=600, bbox_inches='tight')
        plt.show()

    # Save SHAP analysis results
    print("\n8. Saving SHAP analysis results...")

    # Create SHAP importance DataFrame - based on top 10 features sorted by SHAP importance
    shap_importance_df = pd.DataFrame({
        'Feature': [f"Bit_{f}" for f in top_10_features],
        'Original_Index': top_10_features,
        'SHAP_Importance': top_10_shap_importances,
        'Feature_Rank': range(1, len(top_10_features) + 1)
    }).sort_values('SHAP_Importance', ascending=False)

    shap_importance_df.to_csv(os.path.join(submit_dir, 'shap_importance_analysis_top10.csv'), index=False)

    print("SHAP analysis completed successfully!")
    print(f"Analyzed top 10 features (by SHAP importance) from {len(best_features)} final selected features")
    # 9. Save Morgan fingerprint analysis results
    print("\n9. Saving Morgan fingerprint analysis results for final selected features...")

    morgan_analysis_df = pd.DataFrame([
        {
            'Morgan_Bit': f,
            'Importance': morgan_analysis[str(f)]['importance'],
            'Activation_Rate': morgan_analysis[str(f)]['activation_rate'],
            'Activated_Count': morgan_analysis[str(f)]['activated_count'],
            'Total_Molecules': len(morgan_analysis[str(f)]['bit_values']),
            'Sample_SMILES': morgan_substructures.get(f, {}).get('sample_smiles', 'N/A')
        }
        for f in best_features[:10] # Limit to top 10 features
    ])

    morgan_analysis_df.to_csv(os.path.join(submit_dir, 'morgan_detailed_analysis_final_features.csv'), index=False)

    # 4.3. Save molecular substructure information corresponding to Morgan fingerprint bits
    print("\n4.3. Saving molecular substructure information for final selected features...")

    substructure_info = []
    for feature_name in best_features[:10]: # Limit to top 10 features
        if feature_name in morgan_substructures:
            info = morgan_substructures[feature_name]
            substructure_info.append({
                'Morgan_Bit': feature_name,
                'Sample_SMILES': info['sample_smiles'],
                'Bit_Info_Count': len(info.get('bit_info', {}))
            })

    substructure_df = pd.DataFrame(substructure_info)
    substructure_df.to_csv(os.path.join(submit_dir, 'morgan_substructure_analysis_final_features.csv'), index=False)
    print(f"\nMorgan fingerprint analysis completed! Results saved to {submit_dir}")
    print("Total analysis steps: 5")

    return morgan_analysis

# Call visualization function in final evaluation section
if best_model is not None:
    # Call visualization function
    morgan_analysis = visualize_ecfp_importance(
        final_model, best_features, final_feature_importances,
        X_train_raw, X_test_raw, y_train, submit_dir, best_params
    )




