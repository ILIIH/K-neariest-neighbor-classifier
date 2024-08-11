import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

class knn_classifier:      

    def compute_distances_two_loops(self, train_x, train_y, test_x, test_y):
        rows = len(test_x)
        cols = len(train_x)

        result = [[] for _ in range(rows)]
        for i in range(rows):
            result[i] = [None] * cols  
  
        for i in range(rows ):
            for j in range(cols):
                result[i][j] = abs(train_x[i]-test_x[j])+ abs(train_y[i]-test_y[j])

        return result
    

    def knn_cross_val_best_k(distances, labels, k_range, n_splits=5):
        n_samples = distances.shape[1]
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        best_k = None
        best_f1 = -np.inf
        
        for k in k_range:
            fold_f1_scores = []
            
            for train_index, val_index in kf.split(np.arange(n_samples)):
                train_indices = np.array(train_index)
                val_indices = np.array(val_index)
                
                X_train = distances[:, train_indices]
                X_val = distances[:, val_indices]
                y_train = labels[train_indices]
                y_val = labels[val_indices]
                
                # Train KNN classifier
                clf = KNeighborsClassifier(n_neighbors=k, metric='precomputed')
                clf.fit(X_train.T, y_train)  # Note: X_train.T because sklearn expects samples as rows
                
                # Predict and evaluate
                y_pred = clf.predict(X_val.T)
                fold_f1 = f1_score(y_val, y_pred, average='weighted')
                fold_f1_scores.append(fold_f1)
            
            avg_f1 = np.mean(fold_f1_scores)
            print(f"Average F1 Score for k={k}: {avg_f1}")
            
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_k = k
        
        return best_k, best_f1
