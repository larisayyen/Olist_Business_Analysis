
import numpy as np
import pandas as pd
from Olist_Business_Analysis.data_preparation import Olist
from Olist_Business_Analysis.orders_analysis import Order

data = Olist().get_data()

#aggregate the average net_promoter_score
def promoter_score(x):
    return 1 if x==5 else 0 if x==4 else -1

orders = Order().data_trained()
nps = orders.review_score.map(promoter_score).mean()

#one-line code
nps = orders.review_score.map(lambda x: 1 if x == 5 else 0 if x == 4 else -1).mean()

#aggregate the average review score per state
order = data['orders'].copy()
customer = data['customers'].copy()
review = data['order_reviews'].copy()

df = order.merge(review,on='order_id').merge(customer,on='customer_id')
df.groupby('customer_state').agg({'review_score':'mean'})

#aggregate the nps per state

def state_promoter(series):
    return series.map(promoter_score).mean()

df = df.groupby('customer_state').agg({
    'review_score':[np.mean,state_promoter],
    'customer_zip_code_prefix':pd.Series.count
})

df.to_csv('raw_data/nps_state.csv')
