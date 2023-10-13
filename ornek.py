import numpy as np

class RidgeRegScratch():
  # include solver parameter for flexible implementation of gradient descent
  # solution in future, alpha is used in place of lambda to mimic scikit-learn
  def __init__(self, alpha=1.0, solver='closed'):
      self.alpha = alpha
      self.solver = solver

  def fit(self, X, y):
      X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
      self.X_intercept = X_with_intercept
      if self.solver == 'closed':
          # number of columns in matrix of X including intercept
          dimension = X_with_intercept.shape[1]
          # Identity matrix of dimension compatible with our X_intercept Matrix
          A = np.identity(dimension)
          # set first 1 on the diagonal to zero so as not to include a bias term for
          # the intercept
          A[0, 0] = 0
          # We create a bias term corresponding to alpha for each column of X not
          # including the intercept
          A_biased = self.alpha * A
          thetas = np.linalg.inv(X_with_intercept.T.dot(
              X_with_intercept) + A_biased).dot(X_with_intercept.T).dot(y)
      self.thetas = thetas
      return self

  def score(self,y,test_y):
    pass

  def predict(self, X):
      thetas = self.thetas
      X_predictor = np.c_[np.ones((X.shape[0], 1)), X]
      self.predictions = X_predictor.dot(thetas)
      return self.predictions




listeX1 = [1,2,3,4,7,8,9,10]
listeX2 = [2,4,6,7,8,10,12,13]
listeX3 = [1,2,4,3,1,2,4,3]
listeY = [3,9,11,15,17,21,25,27]
X = np.array([[*listeX1],[*listeX2],[*listeX3]]).T
y = np.array(listeY)
print(X,y)

model = RidgeRegScratch(alpha=0.01)
model.fit(X,y)
listeX1 = [6]
listeX2 = [7]
listeX3 = [1]
X_test = np.array([[*listeX1],[*listeX2],[*listeX3]]).T
print(model.predict(X_test))
