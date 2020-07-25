# 数据处理
# 1) X 不需要intercept x0

# linear svm
# 1) c is the parameter decide how much attention we pay on noise
# 2) sklearn.svm.linearSVC give the linear decision boundary
# 3) it predicts confidence scores for samples and mean accuracy (score)
# 4) plot, use Confidence score as c

# kernel
# 1) gaussian kernel function (rbf kernel) =  np.exp(- np.power(x1 - x2, 2).sum() / (2 * (sigma ** 2)))
# 2) sns plot differnet types of y (0/1) automatically
# 3) sklearn.svm.SVC non linear
# 4) predict_prob return a (m,k) prediction, read cols as c to plot

# parameter
# 1) load train and cv
# 2) prepare candidate list
# 3) use GridSearchCV.fit to find the optimal combination
# 4) use GridSearchCV.predict to return the ypred, input cv data
# 5) ypred compare with cv[y] as test

# Spam
# import LogisticRegression to fit logistic regression model

