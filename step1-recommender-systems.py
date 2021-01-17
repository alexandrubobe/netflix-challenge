import numpy as np
import pandas as pd
import os.path
from random import randint

# -*- coding: utf-8 -*-
"""
### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

#Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = './data/submission.csv'
#example_file = './example.csv'
pearson_file = './pearson.csv'
predict = './predict.csv'


# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)
pearson_description = pd.read_csv(pearson_file,delimiter=';',header=None)
predict_description = pd.read_csv(predict, delimiter =';',header = None )
#example_description = pd.read_csv(example_file,delimiter=';', header=None)
#####
##
## COLLABORATIVE FILTERING
##
#####

    
def predict_collaborative_filtering(movies, users, ratings, predictions):
    
    
    # example = np.zeros((4,7))
    # example[0] = [4,0,0,5,1,0,0]
    # example[1] = [4,5,4,0,0,0,0]
    # example[2] = [0,0,0,2,4,5,0]
    # example[3] = [0,3,0,0,0,0,4]
    
    # ex1 = np.zeros((4,7))
    # ex1[0] = [4,0,0,5,1,0,0]
    # ex1[1] = [4,5,4,0,0,0,0]
    # ex1[2] = [0,0,0,2,4,5,0]
    # ex1[3] = [0,3,0,0,0,0,4]
    # average = np.true_divide(example.sum(1),(example != 0).sum(1))
    # for i in range(0,example.shape[0]):
    #     example[i,np.nonzero(example[i])] -= average[i]
        
    # pear = np.corrcoef(example)
    # matrix_ex_pred = np.zeros((4,7))
    
    # for i in range(0,pear.shape[0]):
    #     sorted = np.argsort(-pear[i])
    #     for j in range(0, ex1.shape[1]):
    #         if(ex1[i][j] == 0):
    #             k = 1
    #             sum_ratings = 0
    #             for l in range(0,sorted.shape[0]):
    #                 if(k == 0):
    #                     break
    #                 if(ex1[sorted[l]][j] != 0):
    #                     k = k - 1
    #                     sum_ratings = sum_ratings + ex1[sorted[l]][j]
    #             matrix_ex_pred[i][j] = sum_ratings
    #         else:
    #             matrix_ex_pred[i][j] = ex1[i][j]
    
    # print(matrix_ex_pred)

                    
    #np.savetxt("example.csv", example, delimiter=";")
    #example_pearson = np.corrcoef(example_description)
    
    
    
    # matrix_ratings = np.zeros((users.shape[0]+1, movies.shape[0]+1))
    # matrix_predictions = np.zeros((users.shape[0]+1, movies.shape[0]+1))
    
    # for index, row in ratings.iterrows():
    #     matrix_ratings[row['userID'], row['movieID']] =  row["rating"]
    #     matrix_predictions[row['userID'], row['movieID']] =  row["rating"]
    
    # average = np.true_divide(matrix_ratings.sum(1),(matrix_ratings != 0).sum(1))
    # for i in range(0,matrix_ratings.shape[0]):
    #      matrix_ratings[i,np.nonzero(matrix_ratings[i])] -= average[i]

    #matrix_pearson = np.corrcoef(matrix_ratings)
    #np.savetxt("pearson.csv", matrix_pearson, delimiter=";")   
    
    # for i in range(1,matrix_ratings.shape[0]):
    #     sorted = np.argsort(-pearson_description[i])
    #     for j in range(1, matrix_ratings.shape[1]):
    #         if(matrix_ratings[i][j] == 0):
    #             k = 3
    #             sum_ratings = 0
    #             for l in range(0,sorted.shape[0]):
    #                 if(k == 0):
    #                     break
    #                 if(matrix_predictions[sorted[l]][j] != 0):
    #                     k = k - 1
    #                     sum_ratings = sum_ratings + matrix_predictions[sorted[l]][j]
    #             matrix_predictions[i][j] = sum_ratings / 3
    
    # np.savetxt("predict.csv", matrix_predictions, delimiter=";")


    result = np.zeros((predictions.shape[0],2))
    for i,row in predictions.iterrows():
        result[i] = [i+1,round(predict_description.at[row[0],row[1]])]

    return result.astype(int)


    








    



#####
##
## LATENT FACTORS
##
#####
    
def predict_latent_factors(movies, users, ratings, predictions):
    ## TO COMPLETE

    pass
    
    
#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
  ## TO COMPLETE

  pass


#####
##
## RANDOM PREDICTORS
## //!!\\ TO CHANGE
##
#####
    
#By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]

#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function
predictions = predict_collaborative_filtering(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
   #Formates data
   predictions = [map(str, row) for row in predictions]
   predictions = [','.join(row) for row in predictions]
   predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
    #Writes it dowmn
   submission_writer.write(predictions)
