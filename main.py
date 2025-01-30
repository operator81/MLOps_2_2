import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Iris_Classification')

with mlflow.start_run(run_name='RandomForestClassifier'):
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=5, max_depth=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mlflow.log_param('n_estimators', 5)
    mlflow.log_param('max_depth', 1)

    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric('accuracy', accuracy)

    mlflow.sklearn.log_model(model, 'RandomForestClassifier')
