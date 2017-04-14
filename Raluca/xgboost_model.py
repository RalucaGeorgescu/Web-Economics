import numpy as np
import xgboost
import pandas as pd
import re
import csv

from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

df = pd.read_csv("~/webecon/dataset/train.csv")
df_valid = pd.read_csv("~/webecon/dataset/validation.csv")
df_test = pd.read_csv("~/webecon/dataset/test.csv")

train_split = np.array_split(df, 1)
train = train_split[0]
train_y = train['click']

cols = ['click', 'weekday', 'hour', 'bidid', 'logtype', 'userid', 'useragent',
       'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid',
       'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat',
       'slotprice', 'creative', 'payprice', 'keypage',
       'advertiser', 'usertag', 'bidprice']

train_x = train[cols]


label_encoder = LabelEncoder()
vectorizer = DictVectorizer()

def get_browser(useragent):
    useragent_data = re.split('_',useragent)
    return useragent_data[1]

def get_os(useragent):
    useragent_data = re.split('_',useragent)
    return useragent_data[0]

def bid_request_features(bid,have_click):
    result = defaultdict(float)
    result['weekday=' + str(bid['weekday'].item())] += 1.0
    result['hour=' + str(bid['hour'].item())] += 1.0
    result['userid=' + bid['userid']] += 1.0
    result['broswer=' + get_browser(bid['useragent'])] += 1.0
    result['os=' + get_os(bid['useragent'])] += 1.0
    result['IP=' + bid['IP']] += 1.0
    result['region=' + str(bid['region'].item())] += 1.0
    result['city=' + str(bid['city'].item())] += 1.0
    result['adexchange=' + bid['adexchange']] += 1.0
    result['domain=' + bid['domain']] += 1.0
    result['url=' + bid['url']] += 1.0
    result['slotid=' + bid['slotid']] += 1.0
    result['slotwidth=' + str(bid['slotwidth'].item())] += 1.0
    result['slotheight=' + str(bid['slotheight'].item())] += 1.0
    result['slotvisibility=' + bid['slotvisibility']] += 1.0
    result['slotformat=' + bid['slotformat']] += 1.0
    result['slotprice=' + str(bid['slotprice'].item())] += 1.0
    result['creative=' + bid['creative']] += 1.0
    result['keypage=' + bid['keypage']] += 1.0 
    result['advertiser=' + str(bid['advertiser'].item())] += 1.0

    usertag_data = re.split(',',bid['usertag'])
    for tag in usertag_data:
        result['usertag=' + str(tag)] += 1.0
        
    if have_click == 1:
        result['click=' + str(bid['click'].item())] += 1.0
        
    return result

train_bid_x = vectorizer.fit_transform([bid_request_features(train_x.iloc[i],0) for i in range(len(train_x))])
train_bid_y = label_encoder.fit_transform([train_x.iloc[i]['click'] for i in range(len(train_x))]) 

model = xgboost.XGBClassifier()
model.fit(train_bid_x, train_bid_y)

bid_y= []

def predict_click_probab(test_bids):
    bid_x = vectorizer.transform([bid_request_features(test_bids.iloc[i],0) for i in range(len(test_bids))])
    bid_y.append(model.predict_proba(bid_x.toarray()))
    return bid_y


valid_cols = ['weekday', 'bidid', 'hour', 'userid', 'useragent',
       'IP', 'region', 'city', 'adexchange', 'domain', 'url', 'urlid',
       'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat',
       'slotprice', 'creative', 'keypage',
       'advertiser', 'usertag']

valid_split = np.array_split(df_valid, 1)
valid = valid_split[0]
valid_x = valid[valid_cols]

click_valid_guess = predict_click_probab(valid_x)

valid_x['bidid'] = valid['bidid']
valid_x['pred_click'] = click_valid_guess[0].tolist()

with open("out2.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for i in range(0,len(valid_x)):
        writer.writerow([valid_x['bidid'][i], valid_x['pred_click'][i][0], valid_x['pred_click'][i][1]])
