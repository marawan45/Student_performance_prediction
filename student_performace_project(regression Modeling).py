import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X_train_reg=pd.read_csv("C://Users//maraw//Downloads//my work space//prepared_data//X_train_reg.csv")
y_train_reg=pd.read_csv("C://Users//maraw//Downloads//my work space//prepared_data//y_train_reg.csv").squeeze()
X_test_reg=pd.read_csv("C://Users//maraw//Downloads//my work space//prepared_data//X_test_reg.csv")
y_test_reg=pd.read_csv("C://Users//maraw//Downloads//my work space//prepared_data//y_test_reg.csv").squeeze()



lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)


y_pred = lr.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred)
mae = mean_absolute_error(y_test_reg, y_pred)
r2 = r2_score(y_test_reg, y_pred)

print("Linear Regression Results:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")



# Polynomial Regression

for degree in [2, 3]:
	poly = PolynomialFeatures(degree)
	X_train_poly = poly.fit_transform(X_train_reg)
	X_test_poly = poly.transform(X_test_reg)
	lr_poly = LinearRegression()
	lr_poly.fit(X_train_poly, y_train_reg)
	y_pred_poly = lr_poly.predict(X_test_poly)
	mse_poly = mean_squared_error(y_test_reg, y_pred_poly)
	r2_poly = r2_score(y_test_reg, y_pred_poly)
	print(f"\nPolynomial Regression (degree {degree}) Results:")
	print(f"MSE: {mse_poly:.4f}")
	print(f"R^2: {r2_poly:.4f}")
	joblib.dump(lr_poly, f"poly{degree}_regression_model.pkl")
	joblib.dump(poly, f"poly{degree}_transformer.pkl")
	with open(f"poly{degree}_regression_metrics.txt", "w") as f:
		f.write(f"Polynomial Regression (degree {degree})\nMSE: {mse_poly:.4f}\nR^2: {r2_poly:.4f}\n")
	print(f"Saved degree {degree} polynomial regression model, transformer, and metrics.")

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_reg, y_train_reg)
y_pred_ridge = ridge.predict(X_test_reg)
mse_ridge = mean_squared_error(y_test_reg, y_pred_ridge)
r2_ridge = r2_score(y_test_reg, y_pred_ridge)
print("\nRidge Regression Results:")
print(f"MSE: {mse_ridge:.4f}")
print(f"R^2: {r2_ridge:.4f}")

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_reg, y_train_reg)
y_pred_lasso = lasso.predict(X_test_reg)
mse_lasso = mean_squared_error(y_test_reg, y_pred_lasso)
r2_lasso = r2_score(y_test_reg, y_pred_lasso)
print("\nLasso Regression Results:")
print(f"MSE: {mse_lasso:.4f}")
print(f"R^2: {r2_lasso:.4f}")

#Residual Analysis for Linear Regression
residuals = y_test_reg - y_pred
plt.figure(figsize=(7,4))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted (Linear Regression)')
plt.show()

# Plot regression line (Linear and Polynomial)
plt.figure(figsize=(7,4))
plt.scatter(X_test_reg, y_test_reg, color='blue', alpha=0.3, label='Actual')
plt.scatter(X_test_reg, y_pred, color='black', alpha=0.7, label='Linear Pred')
for degree in [2, 3]:
	poly = PolynomialFeatures(degree)
	X_test_poly = poly.fit_transform(X_test_reg)
	lr_poly = LinearRegression().fit(poly.fit_transform(X_train_reg), y_train_reg)
	y_pred_poly = lr_poly.predict(X_test_poly)
	plt.scatter(X_test_reg, y_pred_poly, alpha=0.5, label=f'Poly deg {degree}')
plt.xlabel('Feature')
plt.ylabel('Total Score')
plt.title('Regression Predictions')
plt.legend()
plt.show()

