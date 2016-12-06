import numpy as np
import matplotlib.pyplot as plt

#this is the amount of samples you will take for greyhounds and labradors
greyhounds = 500
labs = 500

#creating random data-set
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

#plotting the data in form of a histogram
plt.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
plt.show()