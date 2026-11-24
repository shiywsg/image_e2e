import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
import joblib  # [新增] 用于保存 StandardScaler

# 忽略 xgboost 的一些常规版本警告
warnings.filterwarnings("ignore")

def main():
    print("🚀 启动 2-bit 语义信道状态 (S-CSI) 分类器训练与导出框架...\n")
    
    # 1. 读取当前目录下所有包含双轨数据的 CSV 并合并
    all_csv_files = glob.glob("*.csv")
    df_list = []
    
    # 我们仅需要盲估特征和归一化 L2 范数 S-SNR
    features = ['feature_sparsity', 'soft_evm', 'cyclic_error'] # 'physical_snr', 'tar_percent', 'total_variation', 
    target_col = 'norm_ssnr_db'
    required_cols = features + [target_col]
    
    for file in all_csv_files:
        try:
            df_temp = pd.read_csv(file)
            if all(col in df_temp.columns for col in required_cols):
                df_list.append(df_temp)
                print(f"  [读取成功] {file} (包含 {len(df_temp)} 条记录)")
        except Exception as e:
            continue

    if not df_list:
        print("\n❌ 未在当前目录找到有效的数据文件！请确保 CSV 中包含盲估特征和 norm_ssnr_db。")
        return

    df_all = pd.concat(df_list, ignore_index=True)
    df_clean = df_all.replace([np.inf, -np.inf], np.nan).dropna(subset=required_cols)
    
    print(f"\n📊 数据聚合完成！共提取出 {len(df_clean)} 条有效样本。")

    # 2. 核心步骤：将连续的 S-SNR 离散化为 4 个等级 (2-bit)
    print("🗂️ 正在使用 K-Means 算法，根据物理距离自动寻找 4 个自然语义状态的重心...")
    
    # 提取目标列并改变形状为 (n_samples, 1)
    target_data = df_clean[target_col].values.reshape(-1, 1)
    
    # 使用 K-Means 进行 4 分类一维聚类
    kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
    df_clean['scsi_label'] = kbd.fit_transform(target_data).astype(int)
    
    # 提取并打印 K-Means 找到的物理边界
    bins = kbd.bin_edges_[0]
    print("\n[物理聚类状态边界映射表]")
    for i in range(len(bins)-1):
        state_bin_str = f"{i:02b}" # [修正] 4状态对应2-bit
        # 统计落入该自然状态的样本数量
        count = len(df_clean[df_clean['scsi_label'] == i])
        print(f"  状态 {i} ({state_bin_str}) : S-SNR 介于 {bins[i]:>6.2f} dB 到 {bins[i+1]:>6.2f} dB 之间 | 包含样本: {count} 个")

    # 3. 准备训练集和测试集
    X = df_clean[features].values
    y = df_clean['scsi_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}

    # 4. 训练 Random Forest 分类器
    print("\n" + "="*60)
    print("🌲 正在训练 Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results['Random Forest'] = acc_rf

    # 5. 训练 XGBoost 分类器
    print("🚀 正在训练 XGBoost Classifier...")
    xg_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=4, n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42)
    xg_clf.fit(X_train_scaled, y_train)
    y_pred_xgb = xg_clf.predict(X_test_scaled)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    results['XGBoost'] = acc_xgb
    xgb_importance = xg_clf.feature_importances_

    # 6. 打印评估报告
    print("\n" + "█"*60)
    print("🏆 分类器预测准确率 (Accuracy) 排行榜")
    print("█"*60)
    for name, acc in results.items():
        print(f"  [{name:<18}] 准确率: {acc*100:>5.2f}%")
        
    print("\n" + "="*60)
    print("🧠 XGBoost 多分类特征重要性分析")
    print("="*60)
    importance_df = pd.DataFrame({'Feature': features, 'Importance': xgb_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    for idx, row in importance_df.iterrows():
        print(f"  {row['Feature']:<18} : {row['Importance']*100:>5.1f}%")

    print("\n" + "="*60)
    print("📊 XGBoost 分类详细报告 (Precision / Recall / F1-Score)")
    print("="*60)
    print(classification_report(y_test, y_pred_xgb))
    
    # 输出混淆矩阵，帮助分析“是否预测错了，但只错了一个相邻等级？”
    print("\n" + "="*60)
    print("📉 混淆矩阵 (Confusion Matrix)")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred_xgb)
    print(cm)
    print("\n💡 解读提示：对角线上的数字代表完美预测。对角线两侧紧挨着的数字，代表模型预测偏差了 1 个等级（比如把状态 4 错认成了状态 3）。在实际工程中，偏差 1 个等级是可以容忍的！")

    # =========================================================
    # 7. [新增] 模型落盘导出逻辑
    # =========================================================
    print("\n" + "█"*60)
    print("💾 正在导出训练完成的模型文件...")
    print("█"*60)
    
    # 导出标准化器
    scaler_filename = 'scsi_scaler.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"  ✅ 预处理组件已保存: {scaler_filename}")
    
    # 导出 XGBoost 权重
    xgb_filename = 'scsi_xgboost.json'
    xg_clf.save_model(xgb_filename)
    print(f"  ✅ 分类器权重已保存: {xgb_filename}")
    
    print("\n🎉 全部流程结束！现在你可以直接运行接收端程序加载这两个文件了。")

if __name__ == "__main__":
    main()