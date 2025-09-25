from sklearn.model_selection import GridSearchCV
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

X_train=pd.read_csv("C://Users//maraw//Downloads//my work space//prepared_data//X_train.csv")
y_train=pd.read_csv("C://Users//maraw//Downloads//my work space//prepared_data//y_train.csv").squeeze()
X_test=pd.read_csv("C://Users//maraw//Downloads//my work space//prepared_data//X_test.csv")
y_test=pd.read_csv("C://Users//maraw//Downloads//my work space//prepared_data//y_test.csv").squeeze()

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

models = {
	"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
	"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
	"Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
	print(f"\nTraining {name}...")
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f"{name} Accuracy: {acc:.4f}")
	print(f"{name} Classification Report:")
	print(classification_report(y_test, y_pred))

param_grids = {
	"Random Forest": {
		'n_estimators': [100, 200],
		'max_depth': [None, 10, 20],
		'min_samples_split': [2, 5],
		'min_samples_leaf': [1, 2]
	},
	"Logistic Regression": {
		'C': [0.1, 1, 10],
		'solver': ['lbfgs', 'liblinear'],
		'max_iter': [500, 1000]
	},
	"Gradient Boosting": {
		'n_estimators': [100, 200],
		'learning_rate': [0.05, 0.1, 0.2],
		'max_depth': [3, 5, 7]
	}
}

models = {
	"Random Forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
	"Logistic Regression": LogisticRegression(random_state=42, class_weight="balanced"),
	"Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

best_models = {}
for name, model in models.items():
	print(f"\n--- Grid Search for {name} ---")
	grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='f1_macro', n_jobs=-1, verbose=2)
	grid_search.fit(X_train, y_train)
	print(f"Best parameters for {name}: {grid_search.best_params_}")
	best_model = grid_search.best_estimator_
	best_models[name] = best_model
	y_pred = best_model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f"{name} Best Accuracy: {acc:.4f}")
	print(f"{name} Best Classification Report:")
	print(classification_report(y_test, y_pred))
	joblib.dump(best_model, f"best_{name.replace(' ', '_').lower()}_model.pkl")
	print(f"Saved best {name} model as best_{name.replace(' ', '_').lower()}_model.pkl")
