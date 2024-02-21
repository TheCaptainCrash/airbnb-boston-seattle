# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:15:50 2023

@author: Ben
"""
import math
import re
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import normalize


def coef_weights(lm_model, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)

    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df


def find_optimal_lm_mod(X, y, cutoffs, test_size=.30, random_state=42):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        # reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])
        reduce_X = normalize(reduce_X)
        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            reduce_X, y, test_size=test_size, random_state=random_state)

        # fit the model and obtain pred response
        lm_model = LinearRegression()
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        # append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    best_cutoff = max(results, key=results.get)

    # reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    # split the data into train and test
    reduce_X_n = normalize(reduce_X)
    X_train, X_test, y_train, y_test = train_test_split(
        reduce_X_n, y, test_size=test_size, random_state=random_state)

    # fit the model
    lm_model = LinearRegression()
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test, reduce_X


def get_column_names(df):
    return df.columns.values


def impute_mean(df, col):
    df[col].fillna(df[col].mean())


def impute_median(df, col):
    df[col].fillna(df[col].median())


def complex_category_to_dummy(df, col):
    '''
    
    Parameters
    ----------
    df : dataframe
    col : column to index into the dataframe with

    Returns
    -------
    None - dataframe is transformed
    
    Adds dummy columns based off the unique values in multiselect columns
    
    '''
    categories = get_unique_values_from_combinations(df, col)
    col_list = df[col].to_list()
    for cat in categories:
        if cat:
            cat_bool = [cat in x for x in col_list]
            df[cat] = cat_bool
    return df


def get_category_columns(df):
    return df.select_dtypes(include="object").columns.values


def get_numeric_columns(df):
    return df.select_dtypes(include=["int", "float"]).columns.values


def get_unique_values_from_combinations(df, col):
    '''
    
    Parameters
    ----------
    df : dataframe
    col : column to index into the dataframe with

    Returns
    -------
    uniqueValues : set of all possible values in the column.
    Column is expected to contains rows that are multi-select lists from some
    greater set.
    
    i.e.
    colA
    [A,C]
    [B]
    [B,C]
    
    will return the set A,B,C - NOT [A,C],[B],[B,C]

    '''
    uniqueCombos = df[col].unique()
    uniqueValues = set()
    for combo in uniqueCombos:
        if type(combo) != str:
            continue
        values = [x.strip() for x in combo.strip(
            "{").strip("}").strip("[").strip("]").split(",")]
        uniqueValues.update(values)
    return uniqueValues


def convert_column_to_bool(df, col):
    '''

    Parameters
    ----------
    df : dataframe
    col : column to index into the dataframe with

    Returns
    -------
    None - dataframe is transformed
    
    Converts specified columns in dataframe to boolean values. NaNs remain NaN.

    '''
    colVals = df[col]
    cIndex = df.columns.get_loc(col)
    convertedVals = [math.nan] * len(colVals)
    for i, val in enumerate(colVals):
        if isinstance(val, str):  # nans will appear to be non-string values
            convertedVals[i] = val.upper() == 'T' or val.upper() == 'TRUE'
    df.drop(col, inplace=True, axis=1)
    df.insert(cIndex, col, convertedVals)


def convert_percent_to_numeric(df):
    '''
    
    Parameters
    ----------
    df : dataframe

    Returns
    -------
    None - dataframe is transformed
    All columns that have a percentile format (i.e., all non null rows end
    in %) are converted to numeric values

    '''
    possibleColumns = get_category_columns(df)
    for cIndex, col in enumerate(possibleColumns):
        nonNullVals = df.loc[df[col].notnull()][col]
        isPercentage = True
        for val in nonNullVals:
            if val[-1] != "%":
                isPercentage = False
                break
        if isPercentage:
            colVals = df[col]
            convertedVals = [math.nan] * len(colVals)
            for i, val in enumerate(colVals):
                if isinstance(val, str):  # nans will appear to be non-string values
                    convertedVals[i] = float(val.strip("%").replace(",", ""))
            df.drop(col, inplace=True, axis=1)
            df.insert(cIndex, col, convertedVals)


def convert_dollars_to_numeric(df):
    '''
    
    Parameters
    ----------
    df : dataframe

    Returns
    -------
    None - dataframe is transformed
    All columns that have a dollar format (i.e., all non null rows start
    with $) are converted to numeric values

    '''
    possibleColumns = get_category_columns(df)
    for cIndex, col in enumerate(possibleColumns):
        nonNullVals = df.loc[df[col].notnull()][col]
        isMoney = True
        for val in nonNullVals:
            if val[0] != "$":
                isMoney = False
                break
        if isMoney:
            colVals = df[col]
            convertedVals = [math.nan] * len(colVals)
            for i, val in enumerate(colVals):
                if isinstance(val, str):  # nans will appear to be non-string values
                    convertedVals[i] = float(val.strip("$").replace(",", ""))
            df.drop(col, inplace=True, axis=1)
            df.insert(cIndex, col, convertedVals)


def convert_date_to_numeric(df):
    '''
    
    Parameters
    ----------
    df : dataframe

    Returns
    -------
    None - dataframe is transformed
    All columns that have a date format are converted to numeric values
    
    Dates are expected to be written as:
        YYYY-MM-DD
        or
        YYYY/MM/DD

    '''
    possibleColumns = get_category_columns(df)
    # TODO: Support more formats
    dayString = '(3[01]|[12][0-9]|0?[1-9])'
    monthString = '(1[0-2]|0?[1-9])'
    yearString = '([0-9]{4})'
    sepString = '(\/|-)'
    for cIndex, col in enumerate(possibleColumns):
        nonNullVals = df.loc[df[col].notnull()][col]
        isDate = True
        for val in nonNullVals:
            if not re.search(f'{yearString}{sepString}{monthString}{sepString}{dayString}', val):
                isDate = False
                break
        if isDate:
            colVals = df[col]
            year = [math.nan] * len(colVals)
            month = [math.nan] * len(colVals)
            day = [math.nan] * len(colVals)
            for i, val in enumerate(colVals):
                if isinstance(val, str):  # nans will appear to be non-string values
                    dt = datetime.strptime(val, '%Y-%m-%d')
                    year[i] = dt.year
                    month[i] = dt.month
                    day[i] = dt.day
            df.drop(col, inplace=True, axis=1)
            df.insert(cIndex, f"{col}_year", year)
            df.insert(cIndex, f"{col}_month", month)
            df.insert(cIndex, f"{col}_day", day)
