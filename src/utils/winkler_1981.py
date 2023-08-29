import numpy as np

def m_star(sigma_0, mu, one_vec):
    """ mu as column vec"""
    sigma_0_inv = np.linalg.inv(sigma_0)
    temp = np.matmul(one_vec.transpose(), sigma_0_inv)
    num = np.matmul(temp, mu)
    den = np.matmul(temp, one_vec)
    return num / den

def s_star_2(sigma_0, delta_0, mu, one_vec, m_star, k):
    """ mu as column vec"""
    sigma_0_inv = np.linalg.inv(sigma_0)
    temp0 = (m_star*one_vec - mu).transpose()
    temp1 = np.matmul(temp0, sigma_0_inv)
    num = delta_0 + np.matmul(temp1, mu)
    temp2 = np.matmul(one_vec.transpose(), sigma_0_inv)
    den = (delta_0 + k - 1)*np.matmul(temp2, one_vec)
    return num / den

def var_consensus(sigma_0, delta_0, mu, one_vec, m_star, K):
    return (delta_0 + K - 1)*s_star_2(sigma_0, delta_0, mu, one_vec, m_star, K)/(delta_0 + K - 3)

def deg_of_freedom(delta_0, k):
    return delta_0 + k - 1