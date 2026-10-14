# 实验一 新闻文本十分类
10243330403 戴贝妮

## 1. 准备数据

原始数据由课程提供（`data-Project1`），本项目不重复上传。运行前请手动放置：

```
data/train_data.csv            含 text、target 两列，7368 条
data/test_data_unlabeled.csv   仅 text 列，2457 条
```

## 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

实际运行版本（结果基于此环境得出）：

| 库 | 版本 |
| --- | --- |
| Python | 3.10 及以上 |
| scikit-learn | 1.9.1 |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| matplotlib | 3.11.2 |

## 3. 运行顺序

```bash
python prepare_data.py                          # 划分 + 清洗
python tfidf_features.py                        # 基线特征
python baseline_svm.py                          # 基线模型
python model_comparison.py                      # 三模型对比
python tune_svm.py                              # 第一轮调参
python svm_second_round.py                      # 第二轮调参
python final_linear_svc_tuning.py               # 精确调参
python linear_svc_char_feature_experiment.py    # 字符特征融合
python logistic_regression_quick_test.py        # 逻辑回归调参
python mlp_experiment.py                        # MLP 与损失曲线
python learning_curve_experiment.py             # 学习曲线
python final_svc_validation_analysis.py         # 最终验证与图表
python final_predict_linear_svc.py              # 生成提交文件
```

全部脚本均从 `data/` 读取、向 `results/` 写入，不使用绝对路径，换机器可直接运行。完整跑一遍约 9 分钟。

## 4. 各脚本职责

| 脚本 | 作用 |
| --- | --- |
| `prepare_data.py` | 固定种子 42 分层划分 5894 训练 / 1474 验证，再分别清洗（先划分后清洗） |
| `tfidf_features.py` | 学习 5000 维词级 TF-IDF，供基线与 MLP 对照使用 |
| `baseline_svm.py` | LinearSVC 基线，只用验证集评估 |
| `model_comparison.py` | 朴素贝叶斯、逻辑回归、LinearSVC 的多组超参对比 |
| `tune_svm.py` | LinearSVC 第一轮网格搜索 |
| `svm_second_round.py` | LinearSVC 第二轮网格搜索 |
| `final_linear_svc_tuning.py` | LinearSVC 最后一轮精确调参 |
| `linear_svc_char_feature_experiment.py` | 词级与字符级 TF-IDF 融合对比，确定权重 0.25 |
| `logistic_regression_quick_test.py` | 逻辑回归调参（20 组网格），最佳 Macro-F1 0.8821 |
| `mlp_experiment.py` | `partial_fit` 逐轮训练 MLP，记录每轮训练与验证损失 |
| `learning_curve_experiment.py` | 训练样本量对 Macro-F1 的影响 |
| `final_svc_validation_analysis.py` | 复现最终结果，输出混淆矩阵、验证曲线、模型对比图 |
| `final_predict_linear_svc.py` | 方案锁定后拟合最终模型，生成提交文件 |

## 5. 实验结果（复现应当得到的数值）

| 项目 | 数值 |
| --- | --- |
| 数据划分 | 训练 5894 / 验证 1474 / 测试 2457 |
| 最终模型验证集 | Accuracy 0.8948、Macro-F1 0.8950 |
| 朴素贝叶斯 | 0.8352 |
| MLP | 0.8442 |
| 逻辑回归 | 0.8821 |
| LinearSVC（词级） | 0.8922 |
| LinearSVC（词级 + 字符级） | 0.8950 |
| 验证曲线 Macro-F1 | C=0.05 → 0.8659，C=1.0 → 0.8950，C=10 → 0.8864 |
| 学习曲线 CV Macro-F1 | 471 条 → 0.696，4715 条 → 0.818 |
| MLP 损失 | 训练 0.029，验证最低 0.469（第 21 轮），结束 0.493 |
| 提交文件 | `predictions.csv`，2457 行 |

## 6. 注意事项

- 数据划分、LinearSVC、MLP 与交叉验证等均使用同一种子 `random_state=42`。
- 数据集划分采用 `train_test_split(stratify=..., random_state=42)`，类别比例在各子集中保持一致。

## 7. 目录结构

```
├── predictions.csv   ← 最终提交文件
├── results/          ← 图表、指标 JSON
├── *.py              ← 13 个实验脚本
├── README.md
```

运行时还会生成 `data/`，其中的划分结果与 TF-IDF 矩阵均由脚本重新生成，只有 `train_data.csv` 与 `test_data_unlabeled.csv` 需按第前文所述自行放入。
