Этот проект создан для тестирования базового функционала MLflow.

Было проведено 5 экспериментов с двумя моделями (RandomForestClassifier и LogisticRegression) и различными гиперпараметрами на датасете 'Iris Flower Classification':
1. RandomForestClassifier, max_depth=1, n_estimators=5 дает метрику accuracy 0.633
2. RandomForestClassifier, max_depth=1, n_estimators=10 дает метрику accuracy 0.9
3. RandomForestClassifier, max_depth=5, n_estimators=100 дает метрику accuracy 1
4. LogisticRegression, solver='lbfgs', penalty=None дает метрику accuracy 1
5. LogisticRegression, solver='newton-cg', penalty=None дает метрику accuracy 1

Вывод: MLflow работает. Датасет 'Iris Flower Classification' достаточно простой, чтобы получать высокие показатели accuracy даже при небольшом числе деревьев и их глубине при использовании RandomForest, а так же при использовании логистической регрессии с алгоритмами оптимизации для мультиномиальных классов.
