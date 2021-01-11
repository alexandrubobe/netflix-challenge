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


# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)

#####
##
## COLLABORATIVE FILTERING
##
#####

def predict_collaborative_filtering(movies, users, ratings, predictions):
   matrix_ratings  = np.zeros((users.shape[0]+1, movies.shape[0]+1))
    
    
    for index, row in ratings.iterrows():
        matrix_ratings[row['userID'], row['movieID']] =  row["rating"]
    
    
    for i in range(1,matrix_ratings.shape[0]):
        sum = 0
        len = 0
        for j in range(1, matrix_ratings.shape[1]):
            if(matrix_ratings[i][j] != 0):
                sum = sum + matrix_ratings[i][j]
                len = len + 1
        if(sum > 0):
            mean = sum / len
            for j in range(1, matrix_ratings.shape[1]):
                if(matrix_ratings[i][j] != 0):
                    matrix_ratings[i][j] = matrix_ratings[i][j] - mean    

    # This part might work but it is very slow!!!
    # Until here we create the matrix and substract the mean of rows
    
    # matrix_users_similarity = np.zeros((users.shape[0]+1,users.shape[0]+1))
    # for i in range(1, users.shape[0]):
    #     for j in range(1, users.shape[0]):
    #         if(i != j):
    #             sum = 0
    #             sumi = 0
    #             sumj = 0
    #             for k in range(1, matrix_ratings.shape[1]):
    #                 if(matrix_ratings[i][k] != 0 and matrix_ratings[j][k] != 0):
    #                     sum = sum + (matrix_ratings[i][k]*matrix_ratings[j][k])
    #                     sumi = sumi + matrix_ratings[i][k] * matrix_ratings[i][k]
    #                     sumj = sumj + matrix_ratings[j][k] * matrix_ratings[j][k]
    #             sumi = np.sqrt(sumi)
    #             sumj = np.sqrt(sumj)
    #             sum = sum / (sumi * sumj)
    #             matrix_users_similarity[i][j] = sum

    # print(matrix_users_similarity) 
                
    print(matrix_ratings)




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
predictions = predict_random(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)
    
    #Writes it dowmn
    submission_writer.write(predictions)
