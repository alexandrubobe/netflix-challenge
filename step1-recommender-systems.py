import numpy as np
import pandas as pd
import os.path
from random import randint
import random
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
def cosine_similarity(matrix):
    sim_index = np.zeros((matrix.shape[0],matrix.shape[0]))
    for i in range(0,matrix.shape[0]):
        for j in range(0,matrix.shape[0]):
            sim_index[i][j] = np.dot(matrix[i],matrix[j]) / (np.linalg.norm(matrix[i]) * np.linalg.norm(matrix[j]))
    return sim_index

def first_collaborative_filtering(movies, users, ratings, predictions):
    matrix_ratings = np.zeros((users.shape[0]+1, movies.shape[0]+1))
    matrix_predictions = np.zeros((users.shape[0]+1, movies.shape[0]+1))
    
    for index, row in ratings.iterrows():
        matrix_ratings[row['userID'], row['movieID']] =  row["rating"]
        matrix_predictions[row['userID'], row['movieID']] =  row["rating"]
    
    
    
    average = np.true_divide(matrix_ratings.sum(1),(matrix_ratings != 0).sum(1))
    for i in range(0,matrix_ratings.shape[0]):
         matrix_ratings[i,np.nonzero(matrix_ratings[i])] -= average[i]

    
    
    matrix_pearson = np.corrcoef(matrix_ratings)
    np.savetxt("pearson.csv", matrix_pearson, delimiter=";")   
    
    
    
    for i in range(1,matrix_ratings.shape[0]):
        sorted = np.argsort(-pearson_description[i])
        for j in range(1, matrix_ratings.shape[1]):
            if(matrix_ratings[i][j] == 0):
                k = 3
                sum_ratings = 0
                for l in range(0,sorted.shape[0]):
                    if(k == 0):
                        break
                    if(matrix_predictions[sorted[l]][j] != 0):
                        k = k - 1
                        sum_ratings = sum_ratings + matrix_predictions[sorted[l]][j]
                matrix_predictions[i][j] = sum_ratings / 3
    
    
    
    np.savetxt("predict.csv", matrix_predictions, delimiter=";")


    
    
    result = np.zeros((predictions.shape[0],2))
    for i,row in predictions.iterrows():
        result[i] = [i+1,predict_description.at[row[0],row[1]]]

    return result


def predict_collaborative_filtering(movies, users, ratings, predictions):
     
    
    matrix_ratings = np.zeros((users.shape[0]+1, movies.shape[0]+1))
    
    for index, row in ratings.iterrows():
        matrix_ratings[row['userID'], row['movieID']] =  row["rating"]
    
    
    
    average = np.true_divide(matrix_ratings.sum(1),(matrix_ratings != 0).sum(1))
    # for i in range(0,matrix_ratings.shape[0]):
    #      matrix_ratings[i,np.nonzero(matrix_ratings[i])] -= average[i]

    matrix_cosine = cosine_similarity(matrix_ratings)
    np.savetxt("cosine.csv", matrix_cosine, delimiter=";")
    
    #matrix_pearson = np.corrcoef(matrix_ratings)
    #np.savetxt("pearson.csv", matrix_pearson, delimiter=";")   
     
   
    #user - user
    # result = []
    # sorted = np.argsort(-pearson_description)
    # for index in range(0, predictions.shape[0]):
    #     user = predictions.at[index,'userID']
    #     movie = predictions.at[index,'movieID']
    #     if(matrix_ratings[user][movie] != 0):
    #         result.append([index+1, matrix_ratings[user][movie]])
    #     else:
    #         sum = 0
    #         len = 0
    #         k = 10
    #         for j in sorted[user]:
    #             if(k == 0):
    #                 break
    #             if(matrix_ratings[sorted[user][j]][movie] != 0):
    #                 sum += pearson_description.at[user,sorted[user][j]] * matrix_ratings[sorted[user][j]][movie]
    #                 len += np.abs(pearson_description.at[user,sorted[user][j]])
    #                 k -= 1
    #         if(len != 0):
    #             if(np.isnan(average[user] + sum/len)):
    #                 if(np.isnan(average[user])):
    #                     result.append([index+1,3])
    #                 else:
    #                     result.append([index+1,average[user]])
    #             else:    
    #                 result.append([index+1,average[user] + sum/len])
    #         if(len == 0):
    #             if(np.isnan(average[user])):
    #                 result.append([index+1,3])
    #             else:
    #                 result.append([index+1,average[user]])
            
    # return result
    

    def predict_item_item_collaborative_filtering(movies, users, ratings, predictions):
    
    matrix_ratings = np.zeros((movies.shape[0]+1, users.shape[0]+1))
    for index, row in ratings.iterrows():
        matrix_ratings[ row['movieID'], row['userID']] =  row["rating"]
    
    average = np.true_divide(matrix_ratings.sum(1),(matrix_ratings != 0).sum(1))


    matrix_cosine = cosine_similarity(matrix_ratings)
    np.savetxt("cosine.csv", matrix_cosine, delimiter=";")
    
    result = []
    sorted = np.argsort(-cosine_description)
    for index in range(0, predictions.shape[0]):
        user = predictions.at[index,'userID']
        movie = predictions.at[index,'movieID']
        if(matrix_ratings[movie][user] != 0):
            result.append([index+1, matrix_ratings[movie][user]])
        else:
            sum = 0
            len = 0
            k = 15
            for j in sorted[movie]:
                if(k == 0):
                    break
                if(matrix_ratings[sorted[movie][j]][user] != 0):
                    sum += cosine_description.at[movie,sorted[movie][j]] * matrix_ratings[sorted[movie][j]][user]
                    len += cosine_description.at[movie,sorted[movie][j]]
                    k -= 1
            if(len != 0):
                if(np.isnan(sum/len)):
                    if(np.isnan(average[movie])):
                        result.append([index+1,3])
                    else:
                        result.append([index+1,average[movie]])
                else:    
                    result.append([index+1, sum/len])
            if(len == 0):
                if(np.isnan(average[movie])):
                    result.append([index+1,3])
                else:
                    result.append([index+1,average[movie]]) 


    return result



#####
##
## LATENT FACTORS
def predict_latent_factors(movies, users, ratings, predictions):
# Utility matrix 
    matrix_ratings = np.zeros((users.shape[0]+1, movies.shape[0]+1))
    matrix = np.zeros((users.shape[0]+1, movies.shape[0]+1))
    
    for index, row in ratings.iterrows():
        matrix_ratings[row['userID'], row['movieID']] =  row['rating'] 
        matrix[row['userID'], row['movieID']] =  row['rating']
    
    #normalise by user
    average = np.true_divide(matrix_ratings.sum(1),(matrix_ratings != 0).sum(1))
    for i in range(0,matrix_ratings.shape[0]):
         matrix_ratings[i,np.nonzero(matrix_ratings[i])] -= average[i]


    # Q first, then P = s * vh
    Q, s, P2 = np.linalg.svd(matrix_ratings)
    Q = Q[:, :40]
    s = s[:40]
    P1 = np.diag(s)
    #cut the P1 and P2
    P2 = P2[:40]

    P = np.matmul(P1, P2)

    for a in range(10):
        for x in range(1, matrix_ratings.shape[0]):
            for i in range(1, matrix_ratings.shape[1]):
                if matrix_ratings[x, i] != 0: 
                    errors =  2 * (matrix_ratings[x, i] - np.dot(Q[x, :], P[:, i]))
                    Q[x, :] = Q[x, :] + 0.0001 * (errors * P[:, i] - 0.7 * Q[x, :])
                    P[:, i] = P[:, i] + 0.0001 * (errors * Q[x, :] - 0.7 * P[:, i])

    #matrix for predictions
    res = []

    for index, row in predictions.iterrows():
        val = np.dot(Q[row[0]], P[:, row[1]])
        avg = np.sum(matrix[row[0]])/np.count_nonzero(matrix[row[0]])
        val += avg
        if val < 1:
            val = 1
        elif val > 5:
            val = 5
        res.append([index+1, val])
    print(len(res))
    return res
    

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

    return [[idx, random.uniform(1.0, 5.0)] for idx in range(1, number_predictions + 1)]

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
