import random
from collections import defaultdict

upper_limits = range(250,350,1)
click_log = []
impr_log = []

def generate_random_bid(upper_limit):
    bids = []
    for i in range(0,len(valid_x)):
        bid = random.randrange(0,upper_limit,1)
        bids.append(bid) 
    valid_x['bid_pred_rnd'] = bids   
    
valid_x['bidprice'] = valid['bidprice']
valid_x['payprice'] = valid['payprice']
valid_x['click'] = valid['click']

def bidding_strategy(limit):
    budget = 6250
    clicks = 0
    impressions = 0
    for i in range(0,len(valid_x)):
        if budget > 0 or i!=len(valid_x): 
            if valid_x['bid_pred_rnd'][i] > valid_x['bidprice'][i]:
                clicks += valid_x['click'][i]
                impressions += 1 
                budget -= valid_x['payprice'][i]/1000
        else:
            break
    
    click_log.append(clicks)
    impr_log.append(impressions)


for limit in upper_limits:
    generate_random_bid(limit)
    bidding_strategy(limit)
