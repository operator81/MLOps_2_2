import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment('Iris_Classification')

with mlflow.start_run(run_name='LogisticRegression'):
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    model = LogisticRegression(penalty=None, solver='newton-cg')
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mlflow.log_param('penalty', 'None')
    mlflow.log_param('solver', 'newton-cg')

    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric('accuracy', accuracy)

    mlflow.sklearn.log_model(model, 'LogisticRegression')

# Команда для запуска сервера: mlflow server --host 127.0.0.1 --port 5000