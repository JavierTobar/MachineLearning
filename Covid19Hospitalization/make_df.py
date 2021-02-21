import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

PROJ_DIR = os.path.dirname(os.getcwd())


def make_hosp_df(threshold=.60):
	"""
	Makes dataframe containing the hospitalization data, grouped weekly starting on Mondays
	hospitalized_cumulative is the cumulative total at the start of the week
	hospitalized_new is reported as the total new cases reported DURING that week. Some discrepancy btw this number and the opening # at the head of 
		next week bc cases can be + or - while cumulative is always a running total (so i think it's ok)
	"""
	hosp_data_path =  os.path.join(PROJ_DIR, 'data', 'aggregated_cc_by.csv')
	hosp_df = pd.read_csv(hosp_data_path, dtype='object')

	#restrict our analysis to only the US (open_covid_region_code = something from the USA) 
	#so exclude other regions from the hospitalizations data
	# the search regions obtained from the other dataset
	search_regions = np.array(['US-AK', 'US-DC', 'US-DE', 'US-HI', 'US-ID', 'US-ME', 'US-MT',
       'US-ND', 'US-NE', 'US-NH', 'US-NM', 'US-RI', 'US-SD', 'US-VT',
       'US-WV', 'US-WY'], dtype=object)
	hosp_df = hosp_df[hosp_df['open_covid_region_code'].isin(search_regions)].reset_index()
	hosp_df = hosp_df[["open_covid_region_code","region_name","date","hospitalized_cumulative", "hospitalized_new"]] # rename the columns

	# convert the dates to datetime format
	hosp_df['date'] = pd.to_datetime(hosp_df['date'])
	# convert the hospitalized cumulative & new to floats
	hosp_df['hospitalized_cumulative'] = hosp_df['hospitalized_cumulative'].astype(float)
	hosp_df['hospitalized_new'] = hosp_df['hospitalized_new'].astype(float)

	# for the cumulative total
	hosp_cum_counts = hosp_df.groupby(['region_name','open_covid_region_code'] )['hospitalized_cumulative'].count() # group by the regions, count how many total datapoints we have for the region
	hosp_nonzero = hosp_df.groupby(['region_name','open_covid_region_code'])['hospitalized_cumulative'].agg(lambda x: x.ne(0.0).sum()) # counts how many values are nonzero
	hosp_cum_frac_nonzero = (hosp_nonzero / hosp_cum_counts)*100.0 # the proportion of the hospitalized_cumulative that are nonzero

	# for the new cases
	hosp_new_counts = hosp_df.groupby('open_covid_region_code')['hospitalized_new'].count()
	hosp_new_nonzero = hosp_df.groupby(['region_name', 'open_covid_region_code'])['hospitalized_new'].agg(lambda x: x.ne(0.0).sum())
	hosp_new_nonzero = (hosp_new_nonzero / hosp_new_counts)*100.0 # the proportion of the hospitalized_new that are nonzero

	cumulative_thresh = hosp_cum_frac_nonzero > threshold 
	new_thresh = hosp_new_nonzero > threshold

	good_states=cumulative_thresh & new_thresh # get the states that have cumulative + new cases > threshold
	states_list = list(good_states[good_states].index.get_level_values(1))

	# i dropped the actual state from the DF becuase we can always recover it from the region code if necessary
	weekly_df = hosp_df.groupby([pd.Grouper(key='date', freq='W-MON'),'open_covid_region_code']).agg({'hospitalized_cumulative':'first','hospitalized_new': 'sum'})
	weekly_df = weekly_df.loc[(slice(None), states_list), :]

	return weekly_df


def make_search_df(threshold=.60):
	"""
	Returns the dataframe of search data. Dates are weekly beginning on Mondays
	"""
	search_data_path =  os.path.join(PROJ_DIR, 'data', '2020_US_weekly_symptoms_dataset.csv')
	search_df = pd.read_csv(search_data_path, dtype='object')
	search_df['date'] = pd.to_datetime(search_df['date'])
	search_df.drop(columns=["country_region_code",'country_region','sub_region_1','sub_region_1_code'], inplace=True) # dont need these columns (can recover them from the code if needed)


	threshold = threshold*len(search_df) #passing the threshold as parameter now 
	search_df.dropna(axis=1, how='all', inplace=True, thresh=threshold) # drop columns where ALL  values are nan and also those that contain <60% non-Nan values
	final_df = search_df.sort_values(['date', 'open_covid_region_code']).set_index(['date', 'open_covid_region_code'])

	return final_df


def merge_dfs(overlap_only=True, search_thresh=.60, hosp_thresh=.60):
	"""
	overlap_only -> include only the dates that overlap completely for the datasets (2020-03-09 to 2020-09-01)
	
	Merges the two datasets on the weeks. Note that some dates don't overlap (search data starts in Jan. and hospitalization data goes until Oct.) so should prob drop those
	"""
	search = make_search_df(threshold=search_thresh)
	hosp = make_hosp_df(threshold=hosp_thresh)
	
	# had to do this to convert datetimes to strings to be able to merge the data
	search.index = search.index.set_levels(search.index.levels[-2].astype(str), level=-2)
	hosp.index = hosp.index.set_levels(hosp.index.levels[-2].astype(str), level=-2)

	merged = search.merge(hosp,how='outer', left_on=['date','open_covid_region_code'], right_on=['date','open_covid_region_code'])
	
	#usually we will want only the overlapping parts of the dataset. if for some reason no, set it to false (True by default)
	if(overlap_only):
		merged  = merged.loc[(slice('2020-03-09', '2020-09-21'),slice(None))]
	
	return merged



def normalize_and_scale_df(df, how):
	""" Takes the raw data (from merge_df) and returns it normalized, EITHER according to grouping by regions OR by date.
	Enter "how" as the argument for what type of data you need returned.

	df  = the regular merged df obtained from the merge_df
	how= "date" or "region" depending on which features we want
	"""
	how_code={
		'date':'date',
		'region':'open_covid_region_code'
	}

	df = df.astype(float) # make sure the df is float values
	df.fillna(0, inplace=True) # make sure nans are replace w/0

	df_norm = df.copy()

	# groups the df appropriately depending on region/date
	if how =='region':
		df_norm = df_norm.reset_index().set_index('open_covid_region_code') #set the index to the region
		datecol = df_norm['date'].unique() # keep this in case we need it

		# normalize the data by region
		df_norm = df_norm.groupby('open_covid_region_code')[['symptom:Angular cheilitis', 'symptom:Aphonia',
       'symptom:Burning Chest Pain', 'symptom:Crackles',
       'symptom:Dysautonomia', 'symptom:Hemolysis', 'symptom:Laryngitis',
       'symptom:Myoclonus', 'symptom:Rectal pain', 'symptom:Rumination',
       'symptom:Shallow breathing', 'symptom:Stridor',
       'symptom:Urinary urgency', 'symptom:Ventricular fibrillation',
       'symptom:Viral pneumonia', 'hospitalized_cumulative',
       'hospitalized_new']].transform(lambda x: (x - x.mean()) / x.std())

     # only difference is which index is being set really.
	elif how =='date':
		df_norm = df_norm.reset_index().set_index('date') #set the index to the date
		regions = datecol = df_norm['open_covid_region_code'] # keep this in case we need it
		# normalize the data by date
		df_norm = df_norm.groupby('date')[['symptom:Angular cheilitis', 'symptom:Aphonia',
       'symptom:Burning Chest Pain', 'symptom:Crackles',
       'symptom:Dysautonomia', 'symptom:Hemolysis', 'symptom:Laryngitis',
       'symptom:Myoclonus', 'symptom:Rectal pain', 'symptom:Rumination',
       'symptom:Shallow breathing', 'symptom:Stridor',
       'symptom:Urinary urgency', 'symptom:Ventricular fibrillation',
       'symptom:Viral pneumonia', 'hospitalized_cumulative',
       'hospitalized_new']].transform(lambda x: (x - x.mean()) / x.std())


    # initialize standard scaler, normalize the data w.r.t regions/dates
	scaler_norm = StandardScaler()
	scaled_features_norm = scaler_norm.fit_transform(df_norm.values)
	
	norm_scaled_df = pd.DataFrame(scaled_features_norm, index=df_norm.index, columns=df_norm.columns)

	norm_scaled_df = norm_scaled_df.sort_values(by=how_code[how]) # use the how_code dict to look up either date or region 
	norm_scaled_df.fillna(0, inplace=True)

	# Reformat the dataframe to be compatible with the rest of the codebase
	if how == 'date':
		norm_scaled_df.insert(0, 'region', np.resize(regions, (464,)))
	elif how == 'region':
		norm_scaled_df.insert(0, 'date', np.resize(datecol, (464,)))
		norm_scaled_df = norm_scaled_df.sort_values(['date', 'open_covid_region_code'])



	return norm_scaled_df, how # returns the normalized df and how it was scaled (date or region) in case you need that argument


def main():
#	df1 = make_hosp_df()
#	df2 = make_search_df()
	#print(df1)
	#print(df2)
	merge = merge_dfs()
	#merge.to_csv(os.path.join(PROJ_DIR, 'data', 'merged_data.csv'))
	#print(merge)
	scaled_date, _ = normalize_and_scale_df(merge,'date')
	scaled_rgn,_ = normalize_and_scale_df(merge, "region")

#	print(scaled_date.head())
#	print(scaled_rgn.head())
	print(scaled_date.describe())
	print('rgn')
	print(scaled_rgn.describe())



if __name__== '__main__':
	main()


