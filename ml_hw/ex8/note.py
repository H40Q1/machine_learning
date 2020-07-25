# anomaly detection
# 1) split data into train cv test, sklearn.model_selection has train_test_split
# 2) ravel y
# 2.5) stats.multivariate_normal build model by input mu and co
# 3) use X train to calcuate mu and cov, build model, and find best epsilon by input cv data
# 4) use X train and CV to calculate mu and cov, build model and input X test to see the prediction
# 5) sklearn.metrics has f1_score and classification_report


# 推荐系统
# 1) 处理数据时 ravel x 和 theta 为promater方便计算regularized term
# 2）优化时args用y norm = y - mean
# 3）预测值再加上 mean

