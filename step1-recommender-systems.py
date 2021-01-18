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
    sim_index = []
    for row1 in matrix:
        for row2 in matrix:
            sim_index.append(sum(row1*row2)/np.sqrt(sum(row1**2) * sum(row2**2)))
    return np.array(sim_index).reshape((matrix.shape[0],matrix.shape[0]))


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
    
    
    # example = np.zeros((4,7))
    # example[0] = [4,0,0,5,1,0,0]
    # example[1] = [4,5,4,0,0,0,0]
    # example[2] = [0,0,0,2,4,5,0]
    # example[3] = [0,3,0,0,0,0,4]
    
    # ex_pre = np.zeros((3,2))
    # ex_pre[0] = [1,3]
    # ex_pre[1] = [2,2]
    # ex_pre[2] = [3,3]
    # average = np.true_divide(example.sum(1),(example != 0).sum(1))
    # for i in range(0,example.shape[0]):
    #     example[i,np.nonzero(example[i])] -= average[i]
        
    # pear = np.corrcoef(example)
    # matrix_ex_pred = np.zeros((4,7))

    # ex_result =[]
    # sorted = np.argsort(-pear, axis=1)
    # for index in range(0,ex_pre.shape[0]):
    #     user = int(ex_pre[index][0])
    #     movie = int(ex_pre[index][1])
    #     if(example[user][movie]!=0):
    #         ex_result.append([index+1,example[user][movie]])
    #     else:
    #         sum = 0
    #         len = 0
    #         k = 2
    #         for j in sorted[user]:
    #             if(k == 0):
    #                 break
    #             print(example[user][sorted[user][j]])
    #             if(example[user][sorted[user][j]] != 0):
    #                 sum += abs(pear[user][j] * example[user][sorted[user][j]])
    #                 len += abs(pear[user][j])
    #                 k -= 1
           
    #         if(len != 0):
    #             ex_result.append([index+1,sum/len])
    #         if(len == 0):
    #             ex_result.append([index+1,3])
            
    # print(ex_result)


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
    
    # ex_result =[]
    # sorted = np.argsort(-pear, axis=1)
    # for index in range(0,ex_pre.shape[0]):
    #     user = int(ex_pre[index][0])
    #     movie = int(ex_pre[index][1])
    #     if(example[user][movie]!=0):
    #         ex_result.append([index+1,example[user][movie]])
    #     else:
    #         sum = 0
    #         len = 0
    #         k = 2
    #         for j in sorted[user]:
    #             if(k == 0):
    #                 break
    #             print(example[user][sorted[user][j]])
    #             if(example[user][sorted[user][j]] != 0):
    #                 sum += abs(pear[user][j] * example[user][sorted[user][j]])
    #                 len += abs(pear[user][j])
    #                 k -= 1
           
    #         if(len != 0):
    #             ex_result.append([index+1,sum/len])
    #         if(len == 0):
    #             ex_result.append([index+1,3])
    
    # 
   
    
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
