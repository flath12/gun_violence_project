import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', 500)

#load data files
data = pd.read_csv('gun-violence-data_01-2013_03-2018_BM.csv', sep=',')
state_gun_index_data = pd.read_csv('Gun Index Data - by State.csv', sep=',')
zip_income_data = pd.read_csv('income by zip.csv', sep=',', dtype={'zip': object})
zip_longlat_data = pd.read_csv('ziplonglat.csv', sep=',', dtype={'ZIP': object})
cvpi_data = pd.read_csv('CPVI lookup_simplified.csv', sep=',')

#isolate columns of interest in main data file
wanted_columns = data.drop(columns=['incident_url','source_url','incident_url_fields_missing','location_description','notes','participant_age_group','participant_name','participant_relationship','participant_status','participant_type','sources','state_house_district','state_senate_district'])

#add state gun index data
wanted_columns = pd.merge(wanted_columns, state_gun_index_data, on = 'state', how='left')

#this was used to look at the various incident characteristics in a separate file. Not needed for actual data processing. 
#with open('incident_chars.csv','w') as file:
#    for line in wanted_columns.incident_characteristics.unique():
#        file.write(str(line))
#        file.write('\n')

#fill gun type column N.A.s with 0:Unknown
wanted_columns['gun_type_parsed'] = wanted_columns['gun_type'].fillna('0:Unknown')
gt = wanted_columns.groupby(by=['gun_type_parsed']).agg({'n_killed': 'sum', 'n_injured' : 'sum', 'state' : 'count'}).reset_index().rename(columns={'state':'count'})

#transalte actual gun types involve into gun categories
handguns = ('30-30 Win','38 Spl','357 Mag','45 Auto','25 Auto','44 Mag','40 SW','380 Auto','Handgun','32 Auto','10mm','9mm')
assault_rifles = ('7.62','[AK-47]','22 LR','30-06 Spr','308 Win','Rifle','300 Win','223 Rem [AR-15]')
shotguns = ('28 gauge','Shotgun','16 gauge','12 gauge','20 gauge','410 gauge')

def parse_gtype_row(row):
    wrds = row['gun_type_parsed'].split("||")
    gtypes = []
    for wrd in wrds:
        wrd = wrd.replace("::",":").replace("|1","")
        gtypes.append(wrd.split(":")[1])
    return gtypes

def if_handgun(row):
    gtypes = parse_gtype_row(row)     
    if set(gtypes) & set(handguns):
        val = 1
    else:
        val = 0
    return val

def if_shotgun(row):
    gtypes = parse_gtype_row(row)     
    if set(gtypes) & set(shotguns):
        val = 1
    else:
        val = 0
    return val

def if_ar(row):
    gtypes = parse_gtype_row(row)     
    if set(gtypes) & set(assault_rifles):
        val = 1
    else:
        val = 0
    return val

with_gun_categories = wanted_columns
with_gun_categories["handgun"] = wanted_columns.apply(if_handgun, axis=1)
with_gun_categories["shotgun"] = wanted_columns.apply(if_shotgun, axis=1)
with_gun_categories["assault_rifle"] = wanted_columns.apply(if_ar, axis=1)
with_gun_categories = with_gun_categories.drop(columns=['gun_type_parsed','gun_type'])

#add stolen or not stolen data columns
def parse_stolen_row(row):
    wrds = row['stolen_cat_parsed'].split("||")
    stolen_cats = []
    for wrd in wrds:
        wrd = wrd.replace("::",":").replace("|1","")
        stolen_cats.append(wrd.split(":")[1])
    return stolen_cats

def if_stolen(row):
    stolen_cats = parse_stolen_row(row)     
    if "Stolen" in stolen_cats:
        val = 1
    else:
        val = 0
    return val

def if_not_stolen(row):
    stolen_cats = parse_stolen_row(row)     
    if "Not-stolen" in stolen_cats and "Stolen" not in stolen_cats:
        val = 1
    else:
        val = 0
    return val

with_stolen_categories = with_gun_categories
with_stolen_categories['stolen_cat_parsed'] = with_stolen_categories['gun_stolen'].fillna('0:Unknown')
with_stolen_categories["stolen"] = with_stolen_categories.apply(if_stolen, axis=1)
with_stolen_categories["not_stolen"] = with_stolen_categories.apply(if_not_stolen, axis=1)
with_stolen_categories = with_stolen_categories.drop(columns=['gun_stolen','stolen_cat_parsed'])

#add gender columns
def parse_gender_row(row):
    wrds = row['gender_parsed'].split("||")
    gender_cats = []
    for wrd in wrds:
        wrd = wrd.replace("::",":").replace("|1","")
        gender_cats.append(wrd.split(":")[1])
    return gender_cats

def if_male(row):
    gender_cats = parse_gender_row(row)     
    if "Male" in gender_cats:
        val = 1
    else:
        val = 0
    return val

def if_female(row):
    gender_cats = parse_gender_row(row)     
    if "Female" in gender_cats:
        val = 1
    else:
        val = 0
    return val

with_gender_categories = with_stolen_categories
with_gender_categories['gender_parsed'] = with_gender_categories['participant_gender'].fillna('0:Unknown')
with_gender_categories["male"] = with_gender_categories.apply(if_male, axis=1)
with_gender_categories["female"] = with_gender_categories.apply(if_female, axis=1)
with_gender_categories = with_gender_categories.drop(columns=['participant_gender','gender_parsed'])

#filter to mass shootings
with_gender_categories["killed_and_injured"] = with_gender_categories["n_killed"] + with_gender_categories["n_injured"]
mass_shootings = with_gender_categories.loc[with_gender_categories["killed_and_injured"] >= 4]

#drop records with no gun record included 

with_gender_categories['n_guns_involved'] = with_gender_categories#['n_guns_involved'].fillna(-1)
with_gun_records = mass_shootings#.loc[with_gender_categories['n_guns_involved'] != -1]

#add participant age, count
def parse_age_row(row):
    if "||" in row['age_parsed']:
        wrds = row['age_parsed'].split("||")
    elif "|" in row['age_parsed']:
        wrds = row['age_parsed'].split("|")
    else:
        wrds = [row['age_parsed']]
    ages = []
    for wrd in wrds:
        if ":" in str(wrd):
            wrd = wrd.replace("::",":")#.replace("|1","")
            ages.append(int(wrd.split(":")[1]))
        else:
            ages.append(int(float(wrd)))
    return ages

def participant_count(row):
    ages = parse_age_row(row)     
    return len(ages)

def participant_average(row):
    ages = parse_age_row(row)     
    return round(float(sum(ages)) / float(len(ages)),1)

with_age_cleaned = with_gun_records
with_age_cleaned['age_parsed'] = with_age_cleaned['participant_age'].fillna('0::0')
with_age_cleaned["participant_count"] = with_age_cleaned.apply(participant_count, axis=1)
with_age_cleaned["participant_avg_age"] = with_age_cleaned.apply(participant_average, axis=1)
with_age_cleaned = with_age_cleaned.drop(columns=['age_parsed','participant_age'])

#add zip code by longitude & latitutde in order to map to income level
def add_zip(row):
    sort_1 = zip_longlat_data.iloc[(zip_longlat_data['LAT']-row['latitude']).abs().argsort()]
    sort_1 = sort_1.iloc[(sort_1['LNG']-row['longitude']).abs().argsort()[:1]]
    return sort_1.ZIP.iloc[0]

with_age_cleaned["zip"] = with_age_cleaned.apply(add_zip, axis=1)

with_income = pd.merge(with_age_cleaned, zip_income_data, on = 'zip', how='left')

#add cvpi data from data file. This data was already processed in excel.
with_cvpi = pd.merge(with_income, cvpi_data, on = 'incident_id', how='left')

#add democrat, extreme_left, extreme_right columns
def dem_rep(row):
    if row['Party of Representative'] == 'Democratic':
        dem = 1
    else:
        dem = 0
    return dem

def extreme_left(row):
    dems = with_cvpi.loc[with_cvpi['Party of Representative'] == 'Democratic']
    dem_avg = dems['PVI #'].mean()
    dem_std_dev = dems['PVI #'].std()
    if row['PVI #'] >= (dem_avg + dem_std_dev) and row['Party of Representative'] == 'Democratic':
        return 1
    else:
        return 0

def extreme_right(row):
    rep = with_cvpi.loc[with_cvpi['Party of Representative'] == 'Republican']
    rep_avg = rep['PVI #'].mean()
    rep_std_dev = rep['PVI #'].std()
    if row['PVI #'] >= (rep_avg + rep_std_dev) and row['Party of Representative'] == 'Republican':
        return 1
    else:
        return 0

with_cvpi["democrat"] = with_cvpi.apply(dem_rep, axis=1)

with_cvpi["extreme_left"] = with_cvpi.apply(extreme_left, axis=1)
with_cvpi["extreme_right"] = with_cvpi.apply(extreme_right, axis=1)

#reorder columns to desired order
removed_inc_chars = with_cvpi#.drop(columns=['incident_characteristics'])
cols_reordered = removed_inc_chars[['incident_id','date','state','city_or_county','address',
                                   'latitude','longitude','zip','congressional_district','n_killed','n_injured','killed_and_injured','n_guns_involved',
                                   'handgun','shotgun','assault_rifle','stolen','not_stolen',
                                   'male','female','participant_count','participant_avg_age','Gun Freedom Score','Mean income (dollars)','democrat','PVI #','extreme_left','extreme_right','incident_characteristics']]

#write file to csv
cols_reordered.to_csv('cleaned_gun_data_061018.csv', sep=',')