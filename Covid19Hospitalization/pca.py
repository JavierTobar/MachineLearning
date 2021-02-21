from make_df import make_search_df, merge_dfs
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

PROJ_DIR = os.path.dirname(os.getcwd())

def calc_pca(df):
    """ To perform pca on the data, already thresholded
    Input:
    ------
    -df: the (already thresholded) dataframe of (search) data you want to perform PCA on.
    
    Returns:
    --------
    num_comps: the # of components used
    exp_var: the variance explained by each PC
    cum_var_exp: the cumulative variance explained by all the PCs
    components: the components of the PCA
    """
    df = df.fillna(0).astype(float) # fill in the na with 0
    pca = PCA()
    X = df.values
    X = StandardScaler().fit_transform(X) # normalize the inputs
    components = pca.fit_transform(X) # perform the PCA and return the transformed componenets
    num_comps = pca.n_components_
    exp_var = pca.explained_variance_ratio_ # get the explained variance by each PC
    cum_var_exp = np.cumsum(exp_var) # cumulative variance explained by the PCs
    return num_comps, exp_var, cum_var_exp, components
    

def scree_plot(num_comps, exp_var, cum_var_exp, filename=None):
    """ Make a scree plot to visualize the variance explained, determine # of components appropriate to keep from the 
    
    Inputs:
    -------
    - num_comps, exp_var, cum_var_exp: Use the variables returned from calc_pca.
    - filename: To save the figure, enter a filename (string) here. Will go in ./figures/"filename"
    """
    raise RuntimeError('use scree_plot2 instead!!!')
    return None
    xs = range(num_comps)
    plt.plot(xs, cum_var_exp, label='cumulative var.')
    plt.step(xs,exp_var, where='post', label='individual var.')
    plt.xticks(xs);
    plt.title("explained variance");
    plt.xlabel("principal component");
    plt.ylabel("% explained variance");
    plt.legend();
    if filename is not None:
        plt.savefig(os.path.join(PROJ_DIR, 'figures',filename))


def scree_plot2(num_comps, exp_var, cum_var_exp, filename=None):
    """ Make a scree plot to visualize the variance explained, determine # of components appropriate to keep from the 
    
    Inputs:
    -------
    - num_comps, exp_var, cum_var_exp: Use the variables returned from calc_pca.
    - filename: To save the figure, enter a filename (string) here. Will go in ./figures/"filename"
    """
    fig = plt.figure()
    ax = fig.gca()
    
    xs = range(num_comps)
    ax.step(xs, cum_var_exp, where='post',label='cumulative var.')
    ax.plot(xs,exp_var,  label='individual var.',marker='o')
    ax.set_xticks(xs)
    fig.suptitle("explained variance")
    ax.set_xlabel("principal component")
    ax.set_ylabel("% explained variance")
    ax.legend()
    if filename is not None:
        plt.savefig(os.path.join(PROJ_DIR, 'figures',filename), dpi=300)

   # fig.show()
    return fig
    



def label_df(df):
    """
    Label the DF with dates/states
    (Returns the input dataframe except with the the dates/states encoded numerically.)
    
    ALSO returns the label_endoder for dates AND states so we can recover the inverse transform if needed.
    
    so it returns df, le_states, le_dates
    """
    dates = df.index.get_level_values(0)
    states = df.index.get_level_values(1)
    
    le_dates = LabelEncoder()
    le_states = LabelEncoder()

    dates_labels = le_dates.fit_transform(dates)
    states_labels = le_states.fit_transform(states)
    
    df['dates_labels'] = dates_labels
    df['states_labels'] = states_labels
    
    df.fillna(0, inplace=True)    
    return df, le_dates, le_states 


def Xplot_pca(df, label, pca_comps, filename=None, threshold=None):
    """DEPRECATED

    Plot the pca wrt. either "dates_labels" or "states_labels"
    label = dates_labels/states_labels
    pca_comps = the pca components
    filename - to save the figure
    threshold - to put the threshold of features if it's not .6
    """
    unique_labels = np.unique(df[label])
    # plot the PCs and label the states
    num_labels = len(unique_labels)
    cmap=plt.cm.get_cmap('rainbow', num_labels)

    fig = plt.figure()
    ax = fig.gca()

    plt.scatter(pca_comps[:,0], pca_comps[:,1],c=df[label],cmap=cmap, alpha=.7)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    if label == 'dates_labels':
        plt.title("Dates (threshold: {})".format(threshold))
    elif label == 'states_labels':
        plt.title("States (threshold: {})".format(threshold))
     
    plt.colorbar()
    if filename is not None:
        plt.savefig(os.path.join(PROJ_DIR, 'figures',filename), dpi=300)
 

def plot_pca(df, label, pca_comps, filename=None, LE=None):
    """Plot the pca wrt. either "dates_labels" or "states_labels"
    label = dates_labels/states_labels
    pca_comps = the pca components
    filename - to save the figure
    LE = label encoder- only need if doing "states_labels" to get the inverse transform for the labels
    """
    unique_labels = np.unique(df[label])
    # plot the PCs and label the states
    num_labels = len(unique_labels)
    cmap=plt.cm.get_cmap('rainbow', num_labels)

    fig = plt.figure()
    ax = fig.gca()

    scatter = plt.scatter(pca_comps[:,0], pca_comps[:,1],c=df[label],cmap=cmap, alpha=.7)
    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    colorbar=plt.colorbar()
    if label == 'dates_labels':
        plt.title("Search Trends PCA grouped by Dates")
      
        
        
    elif label == 'states_labels':
        plt.title("PCA grouped by States")
        inv_names = LE.inverse_transform(unique_labels) # inverse labels for the states (coded as integers)
        colorbar.set_ticks(np.linspace(0, num_labels, num_labels))
        colorbar.set_ticklabels(inv_names)

    
    if filename is not None:
        plt.savefig(os.path.join(PROJ_DIR, 'figures/{}'.format(filename)), dpi=300)



def main():
	#sample usage
	df = make_search_df() # regular search df, threshold = .6
	num_comps, exp_var, cum_var_exp, components = calc_pca(df)
	scree_plot2(num_comps, exp_var, cum_var_exp, filename="scree_plot_final.png")
	#labeled_df,_,le_states = label_df(df)
	#plot_pca(labeled_df, 'dates_labels', components, filename="pca_dates_final.png") # plot pca using the dates as labels (visualize any grouping among trends over dates)
	#plot_pca(labeled_df, 'states_labels', components, filename="pca_states_final.png", LE=le_states) # plot pca using the states as labels 



if __name__ == '__main__':
	main()
