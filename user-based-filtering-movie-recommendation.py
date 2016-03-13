import random
import numpy as np
import pdb
from itertools import combinations
from collections import defaultdict
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("User Based Collaborative Filtering")
sc = SparkContext(conf=conf)

def loadMovieNames():
    movieNames= {}
    with open("ml-100k/u.item") as f:
        for line in f:
            fields= line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames
def parseUserInfo(line):
    '''
    Parse each line of the specified data file, assuming a "|" delimiter.
    Key is user_id, converts each rating to a float.
    '''
    line = line.split()
    return line[0],(line[1],float(line[2]))
def findingUserPairs(movie_id, users_with_rating):
    '''
    For each item, find all user-user pairs combos. (i.e. users with the same item)
    '''
    for user1,user2 in combinations(users_with_rating,2):
        return (user1[0],user2[0]),(user1[1],user2[1])
def cosineSim(user_pair, rating_pairs):
    '''
    For each user-user pair, return the specified similarity measure,
    along with co_raters_count.
    '''
    sum_x, sum_xy, sum_y,x = (0.0, 0.0, 0.0,0)

    for rating_pair in rating_pairs:
        sum_x += np.float(rating_pair[0]) * np.float(rating_pair[0])
        sum_y += np.float(rating_pair[1]) * np.float(rating_pair[1])
        sum_xy += np.float(rating_pair[0]) * np.float(rating_pair[1])

        x += 1

    cos_sim = cosine(sum_xy,np.sqrt(sum_x),np.sqrt(sum_y))
    return user_pair, (cos_sim,x)
def cosine(dot_product,rating1_norm_squared,rating2_norm_squared):
    '''
    The cosine between two vectors
    '''
    num = dot_product
    den = rating1_norm_squared * rating2_norm_squared

    return (num / (float(den))) if den else 0.0
def nearNeigh(user, users_and_sims, n):
    '''
    Sort the movie predictions list by similarity and select the top N related users
    '''
    users_and_sims.sort(key=lambda x: x[1][0],reverse=True)
    return user, users_and_sims[:n]
def topMovieRecommendations(user_id, user_sims, users_with_rating, n):
    '''
    Calculate the top N movie recommendations for each user using the
    weighted sum approach
    '''


    # initialize dicts to store the score of each individual item,
    # since an item can exist in more than one item neighborhood
    t = defaultdict(int)
    sim_s = defaultdict(int)

    for (neigh,(sim,count)) in user_sims:

        # lookup the movie predictions for this similar neighbours
        unscored_movies = users_with_rating.get(neigh,None)

        if unscored_movies:
            for (movie,rating) in unscored_movies:
                if neigh != movie:

                    # update totals and sim_s with the rating data
                    t[neigh] += sim * rating
                    sim_s[neigh] += sim

    # create the normalized list of scored movies
    scored_items = [(total/sim_s[movie],movie) for movie,total in t.items()]

    # sort the scored movies in ascending order
    scored_items.sort(reverse=True)

    # take out the movie score
    ranked_items = [x[1] for x in scored_items]

    return user_id,ranked_items[:n]
def fillMovieNames(movienames):
    '''
    Assigns the movie names to dictionary
    '''
    nameDict = loadMovieNames()
    for mid in movienames:
        mname =  nameDict[mid]
        mid = mname
    return movienames

def keyOfFirstUser(user_pair, movie_sim_data):
    '''
    For each user-user pair, make the first user's id key
    '''
    (user1_id,user2_id) = user_pair
    return user1_id,(user2_id,movie_sim_data)

lines = sc.textFile("file:///SparkCourse/ml-100k/u.data")

movie_user_pairs = lines.map(lambda x: (x.split()[1],(x.split()[0],float(x.split()[2])))).groupByKey()

movie_user_pairs = movie_user_pairs.map(lambda x : (x[0], list(set(x[1]))))

paired_users = movie_user_pairs.filter( lambda p: len(p[1]) > 1)

paired_users =paired_users .map(
        lambda p: findingUserPairs(p[0], p[1])).groupByKey()
user_sim = paired_users.map(
        lambda p: cosineSim(p[0], p[1]))
user_sim=user_sim.map(
        lambda p: keyOfFirstUser(p[0], p[1])).groupByKey()
user_sim=user_sim.map(lambda x : (x[0], list(x[1]))).map(
        lambda p: nearNeigh(p[0], p[1], 3))

user_movie_history = lines.map(parseUserInfo).groupByKey().collect()

user_dict = {}
for (user,movie) in user_movie_history:
    user_dict[user] = movie

    u = sc.broadcast(user_dict)

    '''
    Calculate the top-N item recommendations for each user
        user_id -> [movie1,movie2,movie3,...]
    '''
user_movie_recs = user_sim.map(
        lambda p: topMovieRecommendations(p[0], p[1], u.value, 5)).collect()
nameDict = loadMovieNames()
'''
Display the movie recommendation
'''
result= user_movie_recs
movieList = list()
for r in result:
    (user, pair) = r
    for p in pair:
        movieList.append(nameDict[int(p)])

    print "For user id ",user, "movie recommendations are ", movieList
    del movieList[:]
    print "end of tuple"