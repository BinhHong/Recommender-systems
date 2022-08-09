# Recommender-systems

# 1. Collaborative filtering algorithm:
Situation:
- there are $n_m$ movies, $n_u$ users, each movie has $n$ features and movie at index $i$ has features $x_i$ (so $x_i$ has $n$ components). 
Normally, the columns of features are listed after the columns of users.
- other notations:
  - $r(i,j) = 1$  if user $j$ rates movie $i$
  - $y(i,j)$ is the rating
- user $j$ rates in total of $m_j$ movies. So in column of users $j$, there are $m_j$ ratings, others are still $?$
- which one is first:
  - if we have features of each movie first, and need to guess the left ratings $?$ --> use linear regression --> for each user $j$, we have parameters $w_j, b_j$.
  Cost function for user $j$ is
  $$\min_{w_j,b_j}J(w_j,b_j)=\frac{1}{2m_j} \sum_{i:r(i,j)=1} (w_jx_i+b_j-y(i,j))^2 + \frac{\lambda}{2m_j} \sum_{k=1,..,n} w^2_{j,k}$$
  where $w_j =(w_{j,1},...,w_{j,n})$, with $n$ is the number of features. Normally we can remove $m_j$ in denominator. 
  For all users, just take sum and get 
  $$J(w_1,b_1,...,w_{n_u},b_{n_u})=\frac{1}{2} \sum_{j= 1,..,n_u} \sum_{i:r(i,j)=1} (w_jx_i+b_j-y(i,j))^2 + \frac{\lambda}{2} \sum_{j = 1,.., n_u} \sum_{k=1,..,n} w^2_{j,k}$$
  - conversely, if features $x_i$'s are not given, we can predict what they are, based on the assumption that $w_j,b_j$ are given.
  Cost function for row $i$, that is movie $i$:
  $$J(x_i)= \frac{1}{2} \sum_{j : r(i,j)=1} (w_jx_i+b_j-y(i,j))^2 + \frac{\lambda}{2} \sum_{k=1,..,n} x^2_{j,k}$$
   where $x_i =(x_{i,1},...,x_{i,n})$, with $n$ is the number of features. For all rows, we have
   
  $$J(x_1,...,x_{n_m})= \frac{1}{2} \sum_{i=1,..,n_m} \sum_{j : r(i,j)=1} (w_jx_i+b_j-y(i,j))^2 + \frac{\lambda}{2} \sum_{i=1,..,n_m} \sum_{k=1,..,n} x^2_{j,k}$$
  
  - note that the first term in two cases are exactly the same. So we combine together to get
  $$J(w,b,x)= \frac{1}{2} \sum_{(i,j) : r(i,j)=1} (w_jx_i+b_j-y(i,j))^2 + \frac{\lambda}{2} \sum_{j = 1,.., n_u} \sum_{k=1,..,n} w^2_{j,k}+ \frac{\lambda}{2} \sum_{i=1,..,n_m} \sum_{k=1,..,n} x^2_{j,k}$$
  
 
This is extent to other situation when rating is binary, in this case the function used is sig-moid instead of linear,
and therefore we use the similar cost function as in logistic regression.
