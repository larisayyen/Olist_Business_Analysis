
import pandas as pd
import numpy as np

from Olist_Business_Analysis.utils import *
from Olist_Business_Analysis.data_preparation import Olist
from Olist_Business_Analysis.orders_analysis import Order

data = Olist().get_data()
orders = Order().data_trained().copy()
table = data['orders'].copy()

# Merge to get order and customer_state in a DataFrame
mask_columns = ['order_id', 'customer_id']
order_customer = table.drop_duplicates(subset=mask_columns)[mask_columns]
order_state = order_customer.merge(data['customers'],
                                   on='customer_id')[['order_id', 'customer_state']]
orders = orders.merge(order_state,
                      on='order_id')

# review cost
cost_mapping = {1:100, 2:50, 3:40, 4:0, 5:0}

orders['cost'] = orders['review_score'].map(cost_mapping)
orders['revenue'] = orders['price'] * 0.1

orders.groupby('customer_state')\
      .agg({'dim_is_one_star':'mean',
            'order_id':'count'})\
      .sort_values(by='dim_is_one_star',
                   ascending=False).head()

orders_agg = orders.groupby('customer_state')\
                   .agg({'cost':'sum',
                         'revenue':'sum',
                         'order_id':'count'})\
                   .sort_values(by='cost',
                                ascending=False)

# total cost and revenue
orders_agg['share_total_cost'] =\
    orders_agg['cost'] / orders_agg['cost'].sum()

orders_agg['share_total_revenue'] =\
    orders_agg['revenue'] / orders_agg['revenue'].sum()

orders_agg['ratio'] =\
    orders_agg['share_total_cost'] / orders_agg['share_total_revenue']

orders_agg.sort_values(by='ratio',
                       ascending=False,
                       inplace=True)

orders_agg['cum_share_cost'] = orders_agg['cost'].cumsum() \
                                        / orders_agg['cost'].sum()

orders_agg['cum_share_revenue'] = orders_agg['revenue'].cumsum() \
                                        / orders_agg['revenue'].sum()

orders_agg['rank'] = orders_agg['cum_share_cost'].rank()

orders_agg_melt = orders_agg[['rank',
                          'cum_share_cost',
                          'cum_share_revenue']].melt(id_vars=['rank'],
                                                    value_vars=['cum_share_cost',
                                                                'cum_share_revenue'])
# compute state removed

def recompute_metrics(rank):
    list_states = orders_agg[orders_agg['rank'] <= rank].index.to_list()
    df = orders.query("customer_state!="+str(list_states))
    review_score = df['review_score'].mean()
    n_orders = df.shape[0]
    orders_impact =  n_orders - orders.shape[0]
    share_one_star = df['dim_is_one_star'].sum() / n_orders
    margin_ratio = df['revenue'].sum() / df['cost'].sum()
    return {'rank':rank,
            'states_removed':str(list_states),
            'review_score':review_score,
            'share_one_star':share_one_star,
            'orders_impact':orders_impact,
            'margin_ratio':margin_ratio}

a = {}
for i in np.arange(0,7):
    a[i] = recompute_metrics(i)

df_ = pd.DataFrame(a).T

df_.to_csv('raw_data/state_profit_margin.csv')
