
import numpy


def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                '''
                Regularization to determine the unknown ratings and normalize the rating
                '''
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                '''
                Error value calculation and reducing the error value
                '''
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T



if __name__ == "__main__":
    R = [
         [5,3,0,1],
         [4,0,0,1],
         [1,1,0,5],
         [1,0,0,4],
         [0,1,5,4],
        ]

    R = numpy.array(R)

    N = len(R) # Number of users
    M = len(R[0]) # Number of movies
    K = 2 # Number of movie related information

    #Create random P matrix
    P = numpy.random.rand(N,K)
    print "printing P matrix"
    print P

    #Create random Q matrix
    Q = numpy.random.rand(M,K)
    print("Printing Q matrix")
    print Q

    nP, nQ = matrix_factorization(R, P, Q, K)
    #Final vector with user, movie rating pair
    nR = numpy.dot(nP, nQ.T)
    print("Printing computed matrix")
    print(nR)
    print("printing actual matrix")
    print(R)


