import pandas as pd
from sklearn.model_selection import train_test_split
from make_df import merge_dfs
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from make_df import normalize_and_scale_df


def gen_train_test(norm_df, how):
	if(how == 'region'):
		# if we're splitting by states, using 3 states as test & rest as training gives ~20/80 split. 
		training_states =['US-AK','US-DC','US-DE','US-HI','US-ID','US-ME','US-MT','US-ND','US-NE','US-NH','US-NM','US-RI','US-SD']
		test_states = ['US-VT','US-WV','US-WY']
		X_train = norm_df.loc[(training_states),:'symptom:Viral pneumonia']
		y_train = norm_df.loc[(training_states),'hospitalized_cumulative':]

		X_valid = norm_df.loc[(test_states),:'symptom:Viral pneumonia']
		y_valid = norm_df.loc[(test_states),'hospitalized_cumulative':]
		

	elif (how =='date'):
		X_train = norm_df.loc[:'2020-08-10',:'symptom:Viral pneumonia'].values
		y_train = norm_df.loc[:'2020-08-10','hospitalized_cumulative':'hospitalized_new'].values
		#print(X_time_train_norm.shape, y_time_train_norm.shape)
		X_valid = norm_df.loc['2020-08-17':,:'symptom:Viral pneumonia'].values
		y_valid = norm_df.loc['2020-08-17':,'hospitalized_cumulative':'hospitalized_new'].values
		#print(X_time_valid_norm.shape,y_time_valid_norm.shape)



	return X_train, y_train, X_valid, y_valid


def plot_knn_mse(X_train, y_train, X_valid, y_valid, how, filename=None):
	"""
	Plots the MSE and accuracy R^2 score for KNN
	Also returns the optimal N for knn according to minimal RMSE
	
	filename = what you want the graph to be saved as. Goes into src/figures by default
	"""
	MSEs = []
	scorelist=[]
	for n in range(1, 50):
	    knn = KNeighborsRegressor(n_neighbors = n) # initialize KNN w/this number of neighbors
	    knn.fit(X_train, y_train) # fit it to the training data
	    y_pred = knn.predict(X_valid) # predict on the validation set
	    rmse = mean_squared_error(y_valid, y_pred)
	    scor = knn.score(X_valid,y_valid)
	    MSEs.append(rmse)
	    scorelist.append(scor)
	
	fig1 = plt.figure(1)
	ax1 = fig1.gca()
	ax1.plot(MSEs)
	ax1.set_xlabel("# neighbors")
	ax1.set_ylabel("RMSE")
	ax1.set_title("KNN: Grouped by {}".format(how))
	plt.savefig("../figures/{}".format(filename), dpi=300)

	fig2 = plt.figure(2)
	ax2 = fig2.gca()
	ax2.plot(scorelist)
	ax2.set_xlabel("# neighbors")
	ax2.set_ylabel("accuracy (R^2)")
	ax2.set_title("KNN: Grouped by {}".format(how))
	plt.savefig("../figures/knn_accuracy_{}.png".format(how), dpi=300)

	opt_n_neighbors = np.argmin(rmse)
	return opt_n_neighbors




def main():
	big_df = merge_dfs()
	normdf, howdf = normalize_and_scale_df(big_df, 'date')

	X_train, y_train, X_valid, y_valid  = gen_train_test(normdf, howdf)
	

	plot_knn_mse(X_train, y_train, X_valid, y_valid, howdf, "testknn.png")

if __name__=="__main__":
	main()




