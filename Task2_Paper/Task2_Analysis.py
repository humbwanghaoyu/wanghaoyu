import pandas as pd
import numpy as np
import matplotlib
import os
# 解决环境配置问题：直接保存图片，不弹窗报错
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # 新增：传统模型对比
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score

# ==========================================
# 1. 路径修复与数据加载 (解决可复现性问题)
# ==========================================
# 自动获取脚本所在目录，严禁使用 C:\Users 绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
file_name = "heart_failure_clinical_records_dataset.csv"
file_path = os.path.join(current_dir, file_name)

try:
    df = pd.read_csv(file_path)
    print(f"✅ 数据加载成功！当前路径: {file_path}")
    
    # 新增：缺失值与异常值检查
    if df.isnull().values.any():
        df = df.fillna(df.median())
        print("⚠️ 检测到缺失值，已执行中位数填充。")
except Exception as e:
    print(f"❌ 数据加载失败，请确保 CSV 文件与本 Python 脚本在同一个文件夹下。")
    print(f"具体错误: {e}")
    exit()

# ==========================================
# 2. 数据准备与分层抽样 (提高严谨性)
# ==========================================
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# 修正：改用 stratify=y 执行分层抽样，确保类别分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. 模型对比实验 (补充传统模型)
# ==========================================
# A. 传统机器学习模型 (逻辑回归)
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

# B. 深度学习模型 (MLP)
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train)
mlp_pred = mlp_model.predict(X_test_scaled)

# ==========================================
# 4. 特征重要性分析 (论文必选图)
# ==========================================
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='teal')
plt.title("Feature Importance Analysis (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("📊 产出 1：特征重要性图已保存。")

# ==========================================
# 5. 模型对比表格导出 (结构化结果)
# ==========================================
comparison_data = {
    'Model': ['Logistic Regression (Traditional)', 'MLP Classifier (Deep Learning)'],
    'Accuracy': [accuracy_score(y_test, lr_pred), accuracy_score(y_test, mlp_pred)],
    'AUC': [auc(*roc_curve(y_test, lr_model.predict_proba(X_test_scaled)[:, 1])[:2]),
            auc(*roc_curve(y_test, mlp_model.predict_proba(X_test_scaled)[:, 1])[:2])]
}
results_df = pd.DataFrame(comparison_data)
results_df.to_csv("model_performance_comparison.csv", index=False)
print("📂 产出 2：模型对比结果已保存至 model_performance_comparison.csv")

# ==========================================
# 6. ROC 曲线绘制 (验证闭环)
# ==========================================
fpr, tpr, _ = roc_curve(y_test, mlp_model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', label=f'MLP ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
print("📈 产出 3：ROC 曲线已保存。")

# ==========================================
# 7. 预测示例 (满足修改意见第 7 条)
# ==========================================
sample_patient = X_test_scaled[0:1]
death_prob = mlp_model.predict_proba(sample_patient)[0][1]
print(f"\n🔮 预测示例：针对单例样本，预测死亡概率为: {death_prob:.2%}")

print("\n✨ 代码修正完毕！所有产出已保存至当前目录。")
import json
# 导出 JSON 摘要
summary = results_summary.to_dict(orient='records')
with open(os.path.join(current_dir, "analysis_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)
print("📄 产出 4：分析摘要 JSON 已保存。")