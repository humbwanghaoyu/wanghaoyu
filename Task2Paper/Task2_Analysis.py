import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 核心避坑：解决 Qt 插件报错，直接保存图片不弹窗
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_curve, auc

# ==========================================
# 1. 数据加载与初处理
# ==========================================
# 请确保 csv 文件路径正确（建议使用绝对路径）
file_path = r"C:\Users\LENOVO\Desktop\Medical LLM Challenge\Task2\heart_failure_clinical_records_dataset.csv"

try:
    df = pd.read_csv(file_path)
    print("✅ 数据加载成功！")
    print("\n--- 数据集概览 ---")
    print(df.describe())
except Exception as e:
    print(f"❌ 数据加载失败，请检查路径。错误: {e}")
    exit()

# ==========================================
# 2. 因子检测：相关性热力图 (论文必选图)
# ==========================================
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', fmt=".2f")
plt.title("Correlation Heatmap of Clinical Factors")
plt.savefig("correlation_heatmap.png")
print("\n📊 产出 1：热力图已保存为 correlation_heatmap.png")

# ==========================================
# 3. 数据准备与标准化
# ==========================================
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# 划分数据集（先划分，后标准化，防止信息泄露）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 执行 Z-score 标准化 (对神经网络 MLP 至关重要)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. 模型训练 1：随机森林 (用于特征重要性)
# ==========================================
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)  # 树模型不需要标准化数据

# 提取前三名危险因子
feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_3 = feat_importances.nlargest(3)
print("\n🔥 产出 2：影响死亡的前三名危险因子：")
print(top_3)

# ==========================================
# 5. 模型训练 2：[加分项] MLP 深度神经网络
# ==========================================
# 使用标准化后的数据训练
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp_pred = mlp.predict(X_test_scaled)

print("\n🚀 产出 3：MLP 深度神经网络评估报告：")
print(classification_report(y_test, mlp_pred))

# ==========================================
# 6. 模型评估：绘制 ROC 曲线 (论文必选图)
# ==========================================
y_score = mlp.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
print("📈 产出 4：ROC 曲线已保存为 roc_curve.png")

print("\n✨ 挑战二代码部分运行完毕！请检查文件夹中的图片。")