from sklearn import svm

def calculate_beta_svm(design_matrix, y):
    return svm.SVC().fit(design_matrix, y)