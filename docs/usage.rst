Usage
=====

Install MISTIC from PyPI with ``python -m pip install mistic-svm``. The
distribution installs the import package named ``mistic``.

A typical workflow creates cross-validation splits, configures a precomputed
kernel SVM, and tunes model/kernel parameter pairs::

   from sklearn.svm import SVC
   from mistic import cvSet, kernelWrapper, paramSet, score_svc, svmSet

   splits = cvSet(X, y)
   splits.classification(num_sets=5)

   ensemble = svmSet(
       SVC(kernel="precomputed"),
       splits,
       score_svc().score,
       kernel=kernelWrapper("rbf"),
   )
   ensemble.tune_models([
       paramSet(model={"C": 1.0}, kernel={"gamma": 0.1}),
   ])
   predictions = ensemble.predict(X_new)

