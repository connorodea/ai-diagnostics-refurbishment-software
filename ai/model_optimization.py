from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

# Configure logging
logging.basicConfig(filename='../logs/model_optimization.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

def optimize_model(X_train, y_train):
    try:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        logging.info(f"Best Parameters: {grid_search.best_params_}")
        logging.info(f"Best Score: {grid_search.best_score_}")
        return grid_search.best_estimator_
    except Exception as e:
        logging.error(f"Model Optimization Error: {e}")
        return None

if __name__ == "__main__":
    from ai_module import load_data, preprocess_data, train_model
    data = load_data('../data/diagnostic_data.csv')
    if data is not None:
        X, y = preprocess_data(data)
        if X is not None and y is not None:
            optimized_model = optimize_model(X, y)
            if optimized_model:
                joblib.dump(optimized_model, '../ai/optimized_failure_predictor.pkl')
                logging.info("Optimized model saved successfully.")
