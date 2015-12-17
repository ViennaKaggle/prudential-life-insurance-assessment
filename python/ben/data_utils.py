import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import csv
import datetime
import copy
import time


def convert_to_date(year, month):
    try:
        return datetime.date(int(year), int(month), day=15)
    except ValueError:
        return datetime.date(1900, 1, 1)


def date_from_year_week(year, week):
    try:
        week_str = "%s-%s" % (int(year), int(week))
        return datetime.datetime.strptime(week_str + '-1', "%Y-%W-%w")
    except ValueError:
        return None


def transform_data(sales, stores):
    # TRANSFORM SALES DATA
    if 'Sales' in sales.columns:
        sales.Sales = sales.Sales.astype(np.float64)
    sales.Date = sales.Date.astype(np.datetime64)
    sales['woy'] = sales.Date.dt.weekofyear
    sales['month'] = sales.Date.dt.month
    sales['year'] = sales.Date.dt.year
    sales['DayOfYear'] = sales['Date'].dt.dayofyear
    # sales['DaySinceStart'] = sales['Year'] * 365 + sales['DayOfYear']
    sales['Seasonal_4_sin'] = np.sin(sales.DayOfYear/365*4*2*np.pi)
    #sales['Seasonal_4_cos'] = np.cos(sales.DayOfYear/365*4*2*np.pi)
    #sales['Seasonal_3_sin'] = np.sin(sales.DayOfYear/365*3*2*np.pi)
    #sales['Seasonal_3_cos'] = np.cos(sales.DayOfYear/365*3*2*np.pi)
    sales = pd.get_dummies(sales, columns=['StateHoliday'])
    sales = sales.drop(['DayOfYear'], axis=1)
    if 'Customers' in sales.columns:
        sales = sales.drop(['Customers'], axis=1)

    # TRANSFORM STORES DATA
    stores['CompetitionDistance'] = stores.CompetitionDistance.fillna(stores.CompetitionDistance.max())
    # stores['CompetitionDistance'] = np.log1p(stores.CompetitionDistance)
    stores['CompetitionOpenSinceYear'] = stores.CompetitionOpenSinceYear.fillna(2050)
    stores['CompetitionOpenSinceMonth'] = stores.CompetitionOpenSinceMonth.fillna(1)
    stores['CompetitionOpenSince'] = pd.to_datetime(stores[['CompetitionOpenSinceYear',
        'CompetitionOpenSinceMonth']].apply(lambda s: convert_to_date(s[0], s[1]),axis = 1))
    stores = pd.get_dummies(stores, columns=['StoreType', 'Assortment'])
    stores = stores.drop(['CompetitionOpenSinceMonth',
        'CompetitionOpenSinceYear',
        'Promo2',
        'Promo2SinceWeek',
        'Promo2SinceYear',
        'PromoInterval'
    ], axis=1)

    # MERGE STORES AND SALES DATA
    all_data = pd.merge(sales, stores, on='Store')
    all_data['PostComp'] = (all_data['Date'] > all_data['CompetitionOpenSince']).astype(np.int)
    all_data['CompetitionDistance'] = np.where(all_data.PostComp, all_data.CompetitionDistance, all_data.CompetitionDistance.max())
    all_data['Open'] = all_data.Open.fillna(1)
    all_data = all_data.sort_values('Date').reset_index(drop=True)
    all_data = all_data.drop(['CompetitionOpenSince'], axis=1)

    return all_data


def calc_store_sales_distributions(all_data):
    stores_mean_post = all_data[all_data.PostComp==True][['Store', 'Sales']].groupby('Store').mean()
    stores_mean_pre = all_data[all_data.PostComp==False][['Store', 'Sales']].groupby('Store').mean()
    stores_mean_post['Store'] = stores_mean_post.index
    stores_mean_pre['Store'] = stores_mean_pre.index
    stores_mean_post = stores_mean_post.rename(columns={'Sales': 'Sales_mean_post'})
    stores_mean_pre = stores_mean_pre.rename(columns={'Sales': 'Sales_mean_pre'})
    stores_mean_post['PostComp'] = True
    stores_mean_pre['PostComp'] = False

    stores_std_post = all_data[all_data.PostComp==True][['Store', 'Sales']].groupby('Store').std()
    stores_std_pre = all_data[all_data.PostComp==False][['Store', 'Sales']].groupby('Store').std()
    stores_std_post['Store'] = stores_std_post.index
    stores_std_pre['Store'] = stores_std_pre.index
    stores_std_post = stores_std_post.rename(columns={'Sales': 'Sales_std_post'})
    stores_std_pre = stores_std_pre.rename(columns={'Sales': 'Sales_std_pre'})
    stores_std_post['PostComp'] = True
    stores_std_pre['PostComp'] = False

    results_mean = pd.concat([stores_mean_post, stores_mean_pre], axis=0)
    results_std = pd.concat([stores_std_post, stores_std_pre], axis=0)

    results = pd.merge(results_mean, results_std, on=['Store', 'PostComp'])

    #fill missing pre/post competition values with distribution values from the other
    fillers = []
    for row in results.iterrows():
        store = row[1]['Store']
        if len(results[results.Store==store]) == 1:
            new_series = pd.Series(copy.deepcopy(row[1]))
            new_series['PostComp'] = not new_series['PostComp']
            new_series['Sales_mean_post'] = row[1]['Sales_mean_pre']
            new_series['Sales_mean_pre'] = row[1]['Sales_mean_post']
            new_series['Sales_std_post'] = row[1]['Sales_std_pre']
            new_series['Sales_std_pre'] = row[1]['Sales_std_post']
            fillers.append(new_series)
    results = pd.concat([results, pd.DataFrame(fillers)], axis=0)

    return results


def merge_sales_with_distributions(all_data, dist):
    all_data = pd.merge(all_data, dist, how='left', on=['Store', 'PostComp'])
    all_data['Sales_mean'] = all_data[['Sales_mean_post', 'Sales_mean_pre']].sum(axis=1)
    all_data['Sales_std'] = all_data[['Sales_std_post', 'Sales_std_pre']].sum(axis=1)
    all_data['PostComp'] = all_data['PostComp'].astype(int)
    all_data = all_data.drop(['Sales_mean_post', 'Sales_mean_pre', 'Sales_std_post', 'Sales_std_pre', 'PostComp'], axis=1)
    return all_data


def extend_school_holidays(dataframe, compare_offset, dow):
    """For every dow (day of week) that is a school holiday look back
    compare_offset days and check if that day was a SchoolHoliday"""
    oneday = DateOffset(days=compare_offset)
    tmp = dataframe[['Store']]
    tmp['Date'] = dataframe.Date + oneday
    tmp['SchoolHoliday_m1'] = dataframe.SchoolHoliday
    tmp = pd.merge(dataframe, tmp, how='left', on=['Store', 'Date'])
    tmp['SchoolHoliday_m1'] = tmp.SchoolHoliday_m1.fillna(0).astype(int)
    tmp['SchoolHoliday_m1'] = np.logical_and(tmp.DayOfWeek==dow, tmp.SchoolHoliday_m1==1)
    tmp['SchoolHoliday'] = np.where(np.logical_or(tmp.SchoolHoliday==1, tmp.SchoolHoliday_m1), 1, 0)
    return tmp.drop('SchoolHoliday_m1', axis=1)


def harmonize_school_holidays(dataframe):
    dataframe = extend_school_holidays(dataframe, 1, 6)
    dataframe = extend_school_holidays(dataframe, 2, 7)
    return dataframe

def add_column_for_last_holiday_week(dataframe, exclude_year):
    """If today is a SchoolHoliday and today+7days is not, then today is SchoolHolidayEnding"""
    oneday = DateOffset(days=7)
    tmp = dataframe[['Store']]
    tmp['Date'] = dataframe.Date - oneday
    tmp['SchoolHolidayEnding'] = dataframe.SchoolHoliday
    tmp = pd.merge(dataframe, tmp, how='left', on=['Store', 'Date'])
    tmp['SchoolHolidayEnding'] = tmp.SchoolHolidayEnding.fillna(0).astype(int)
    tmp['SchoolHolidayEnding'] = np.logical_and(tmp.SchoolHoliday==1, tmp.SchoolHolidayEnding==0)
    tmp['SchoolHolidayEnding'] = np.logical_and(tmp.month>=7, tmp.SchoolHolidayEnding)
    tmp['SchoolHolidayEnding'] = np.logical_and(tmp.year<exclude_year, tmp.SchoolHolidayEnding)
    tmp['SchoolHolidayEnding'] = np.where(np.logical_and(tmp.month<=9, tmp.SchoolHolidayEnding), 1, 0)
    return tmp


def load_transformed_data():
    # load training and test set
    train = pd.read_csv('../../data/train.csv', dtype={'StateHoliday': np.str})
    test = pd.read_csv('../../data/test.csv')
    store = pd.read_csv('../../data/store.csv')

    # transform training data
    all_data = transform_data(train, store)
    all_data = harmonize_school_holidays(all_data)
    all_data = add_column_for_last_holiday_week(all_data, 2015)
    #all_data = all_data[all_data.Open==1]  #get rid of all closed days
    store_sales_distributions = calc_store_sales_distributions(all_data)
    all_data = merge_sales_with_distributions(all_data, store_sales_distributions)

    # transform test set
    transformed_test = transform_data(test, store)
    transformed_test = harmonize_school_holidays(transformed_test)
    transformed_test = add_column_for_last_holiday_week(transformed_test, 2050)
    transformed_test = merge_sales_with_distributions(transformed_test, store_sales_distributions)
    test_ids = transformed_test.Id
    transformed_test = transformed_test.reindex_axis(all_data.columns, axis='columns', fill_value=0)
    transformed_test = pd.concat([test_ids, transformed_test], axis=1)


    return all_data, transformed_test


def get_raw_values(dataframe):
    """Returns numpy arrays X and y for features and target values"""
    cols_to_drop = [
        'Sales',
        'Date',
    ]
    if 'Id' in dataframe.columns:
        cols_to_drop.append('Id')
    X = dataframe.drop(cols_to_drop, axis=1).values
    y = dataframe['Sales'].values
    return X, y


def write_submission(y_pred, dataframe):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    s1 = pd.Series(dataframe.Id, name='Id').reset_index(drop=True)
    s2 = pd.Series(y_pred, name='Response')
    results = pd.concat([s1, s2], axis=1)
    results = results.sort_values('Id')
    results.to_csv(path_or_buf='submissions/predictions_%s.csv' % timestr, index=False, quoting=csv.QUOTE_NONNUMERIC)
