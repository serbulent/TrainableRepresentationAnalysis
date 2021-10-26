import numpy as np

def proba_mass_split(y):
    folds=len(y[0])
    # get number of samples (obs) and number of classes
    obs, classes = y.shape
    # get number of positive samples for each class
    dist = y.sum(axis=0).astype('float')
    # Caclulate observation probabilty of each class
    dist /= dist.sum()
    index_list = []
    #create and empty matrix of fold distribution
    fold_dist = np.zeros((folds, classes), dtype='float')
    #create empty list to save sample indexes of folds
    for _ in range(folds):
        index_list.append([])
    #start distributing samples to folds based on their observation probability
    for i in range(obs):
        # Distribute first n sample to each fold (n=number of folds)
        if i < folds:
            target_fold = i
        else:
            '''
            current probabilities of observing a class at a fold. This value is calculated by dividing labels at fold to fold sum.
            For example 4th value at second row represents probability of observing the 2nd label at 4rd fold. To do this we get
            transpose of fold dist and each row stands for number labels at folds then divide them to fold sums. 
            
            array([[0. , 0.5 , 0. , 0. , 0.25],
                   [1 , 0. , 0. , 0.33333, 0.],
                   [0. , 0. , 0. , 0.33, 0.25],
                   [0. , 0. , 0. , 0.33, 0.25],
                   [0. , 0.5 , 1. , 0. , 0.25]])
            '''
            normed_folds = fold_dist.T / fold_dist.sum(axis=1)
            '''
            Difference between current probabilities and observational probabilities based on whole dataset. 
            Transpose of normed_folds gives the original fold x label distribution probs. 
            dist is observation probability of each class in the whole dataset. e.g.,
            
            array([0.17857143, 0.25      , 0.14285714, 0.23214286, 0.19642857])
            
            If a value is close to zero than observed probability of this value is same with the original dataset
            If a value is higher or lower than zero than this is the difference between original and observed probabilities.
            e.g.,
            
            array([[-0.17857143,  0.75      , -0.14285714, -0.23214286, -0.19642857],
                   [ 0.32142857, -0.25      , -0.14285714, -0.23214286,  0.30357143],
                   [-0.17857143, -0.25      , -0.14285714, -0.23214286,  0.80357143],
                   [-0.17857143,  0.08333333,  0.19047619,  0.10119048, -0.19642857],
                   [ 0.07142857, -0.25      ,  0.10714286,  0.01785714,  0.05357143]])
            
            '''
            how_off = normed_folds.T - dist
            '''
            (y[i] - .5).reshape(1, -1) creates an array based of observations. If a class is observed in the sample "i",
            the calculated value will be 0.5 otherwise -0.5 e.g., array([[ 0.5, -0.5,  0.5,  0.5,  0.5]])
            This array will be used for weighting how_off matrix to select the fold that the sample "i" will be assigned.
            
            A high positive value in the how_off matrix means there are high number of samples exits in current state,
            but they are not expected according to original distribution. A high negative number means there are lower number of
            samples exits than expected for that fold&label. 
            
            By using dot product lower values will be acquired if there are lower number of samples exits for a label and the 
            candidate sample annotated by the needed label. Otherwise, higher values will be produced if there are higher number
            of observations than expected and the candidate sample annotated by the unneeded label.           
            
            Also when we get the dot product of weight array with the transpose of how_off we get minimum value
            for the fold which we expect the most probable observation but not found. 
            Using np.argmin we can get the number of this fold and assign our sample to it.
            '''
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)
    #print("Fold distributions are")
    #print(fold_dist)
    return index_list

def split(folds):
    train_index = []
    test_index = []
    cross_val={'train': train_index, 'test': test_index}
    #divide train and test indexes
    for i, testi in enumerate(folds):
        train_index.append(folds[:i] + folds[i+1:])
        test_index.append(testi)
    
    #store indexes in lists
    train_list = []
    for inner_list in train_index:
      inner_train_list = []
      for subinner_list in inner_list:
        inner_train_list.extend(subinner_list)
      train_list.append(inner_train_list)
    train_index = train_list
    return zip(train_index, test_index)

def fibonacci(size):
    fibonacci_numbers = [0, 1]
    for i in range(2,size):
        fibonacci_numbers.append(fibonacci_numbers[i-1]+fibonacci_numbers[i-2])
    return fibonacci_numbers
