# MyFirstMLProject
my first Machine Learning project. No large datasets, just a way to familiarize myself with many of the ML algos that are out there

For this project, the idea is to analyze a dataset of coupon users and determine whether or not someone will use a coupon on a sample set.
I used 6 different ML methods to compare accuracy.
- Perceptron
- Logistic Regression
- Supoort Vector Machine
- Decision Tree Learning
- Random Forest
- K-Nearest Neighbor

### Observations
- Part 1
Correlation analysis was run on the dataset provided by ACC. From the top correlations, one of the best indicators for coupon usage is the distance that the individual/person who holds the coupon is willing to travel to use the coupon. The covariance between coupon usage and expiration has a high magnitude, which could indicate that those two will move with each other, meaning that as coupon expires the usage will increase/decrease. In observing the correlation with those who have used the coupon, the best correlation is with the expiration, along with the destination. Given that these top variables appear to have the highest correlation; I believe that a Machine Learning algorithm will yield an above moderate ability to return a confident prediction on who will use the coupon or not. The amount of data that is available, can be cleaned and scaled in a manner that can meet the needs in ability to achieve the >= 70% coupon suage prediction.

- Part 2
After running the coupon data through various machine learning algorithms, 4 out of the 6 classifications used to test the usage of a coupon is predictable, met the threshold of accuracy of 70%, set by ACC. This ability to better predict who may use a coupon and what factors can help drive the person using the coupon decision. These tests were performed with a large sample size of data and then tested again with a portion of the actual data added back in and retested for accuracy. As stated above, 4 of the 6 algorithms used yielded an accuracy for test sample and combined samples of >= 70%. Therefore, with current features used, we are confident that we can us the benefits of Machine Learning to improve coupon usage and increase overall sales.
