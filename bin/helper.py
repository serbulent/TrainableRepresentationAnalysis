import numpy as np

def proba_mass_split(y, folds=5):
    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(obs):
        if i < folds:
            target_fold = i
        else:
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)
    print("Fold distributions are")
    print(fold_dist)
    return index_list

def split(folds):
    train_index = []
    test_index = []
    cross_val={'train': train_index, 'test': test_index}
    for i, testi in enumerate(folds):
        train_index.append(folds[:i] + folds[i+1:])
        test_index.append(testi)
    
    train_list = []
    for inner_list in train_index:
      inner_train_list = []
      for subinner_list in inner_list:
        inner_train_list.extend(subinner_list)
      train_list.append(inner_train_list)
    train_index = train_list
    return zip(train_index, test_index)

if __name__ == '__main__':
    print("Test stratified k-fold and split functions\n")
    np.random.seed(42)
    y = np.random.randint(0, 2, (20, 5))
    #If a sample was not annotated by any of classes (labels) than this sample will be removed
    y = y[np.where(y.sum(axis=1) != 0)[0]]
    index_list = proba_mass_split(y)
    print("Index list / Folds")
    print(index_list)
    print("\nTrain-Test Folds\n")
    for train, test in split(index_list):
      print("Train")
      print(train)
      print("Test")
      print(test)

