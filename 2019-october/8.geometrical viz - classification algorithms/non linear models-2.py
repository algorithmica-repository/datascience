import sys
sys.path.append("E:/New Folder/utils")

import kernel_utils as kutils
import classification_utils as cutils
from sklearn import model_selection, linear_model, svm, preprocessing, pipeline, neural_network


#2-d classification pattern
X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)
X, y = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=1000, noise=0.1)

cutils.plot_data_2d_classification(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

#perceptron algorithm
stages = [('features', kutils.KernelTransformer('rbf')) ,
          ('clf', linear_model.Perceptron(max_iter=1000))
          ]
perceptron_pipeline = pipeline.Pipeline(stages)
perceptron_pipeline_grid  = {'features__gamma':[0.1, 0.01, 0.2]}
pipeline_object = cutils.grid_search_best_model(perceptron_pipeline, perceptron_pipeline_grid, X_train, y_train)
final_estimator = pipeline_object.named_steps['clf']
print(final_estimator.intercept_)
print(final_estimator.coef_)
cutils.plot_model_2d_classification(pipeline_object, X_train, y_train)

#logistic regression algorithm
stages = [('features', kutils.KernelTransformer('rbf')) ,
          ('clf', linear_model.LogisticRegression())
          ]

lr_pipeline = pipeline.Pipeline(stages)
lr_pipeline_grid  = {'features__gamma':[0.1, 1, 5,10]}
pipeline_object = cutils.grid_search_best_model(lr_pipeline, lr_pipeline_grid, X_train, y_train)
final_estimator = pipeline_object.named_steps['clf']
print(final_estimator.intercept_)
cutils.plot_model_2d_classification(pipeline_object, X_train, y_train)

#linear svm algorithm
kernel_svm_estimator = svm.SVC(kernel='rbf')
kernel_svm_grid = {'gamma':[0.01, 0.1, 1, 2, 5, 10] }
final_estimator = cutils.grid_search_best_model(kernel_svm_estimator, kernel_svm_grid, X_train, y_train)
print(final_estimator.intercept_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

#artifical neural network
ann_estimator = neural_network.MLPClassifier()
ann_grid = {'hidden_layer_sizes':[(3, 4), (10, 20)] }
final_estimator = cutils.grid_search_best_model(ann_estimator, ann_grid, X_train, y_train)
print(final_estimator.intercepts_)
print(final_estimator.coefs_)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)
