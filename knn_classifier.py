
class knn_classifier:      

    def compute_distances_two_loops(self, train_x, train_y, test_x, test_y):
        rows = len(test_x)
        cols = len(test_y)

        result = [[] for _ in range(rows)]
        for i in range(rows):
            result[i] = [None] * cols  
  
        for i in range(cols):
            for j in range(rows):
                print(train_x[i])
                print(test_x[j])

                result[i][j] = abs(train_x[i]-test_x[j])+ abs(train_y[i]-test_y[j])

        return result