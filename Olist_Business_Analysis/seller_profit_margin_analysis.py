
import numpy as np
import pandas as pd
from seller_analysis import Seller

seller = Seller.get_training_data().copy()

# revenue per seller
olist_revenue = seller.revenues.sum()
olist_month_revenue = seller.months_on_olist.sum()*80
olist_sales_revenue = seller.sales.sum()*0.1

# cost of bad reviews
olist_reputation_cost = seller.cost_of_reviews.sum()

# it cost
olist_it_costs_all_orders = 500000

# total cost
olist_total_costs = 500000 + seller.cost_of_reviews.sum()

# profit
olist_gross_profits = seller.profits.sum()
olist_net_profits = olist_gross_profits - olist_it_costs_all_orders

sorted_sellers = seller.sort_values(by='profits')[['profits', 'n_orders', 'revenues']].reset_index()

# Olist's net_profits for various seller cut-offs

# Analysis excluding IT costs
revenues_per_seller_removed = olist_revenue - np.cumsum(sorted_sellers.revenues)
gross_profits_per_seller_removed = olist_gross_profits - np.cumsum(sorted_sellers.profits)

# Add the IT costs of Olist's platform
# IT costs =  A * (n_orders)**0.5

A = olist_it_costs_all_orders / (seller['n_orders'].sum()**0.5)
n_orders_per_seller_removed = sorted_sellers.n_orders.sum() - np.cumsum(sorted_sellers.n_orders)
it_costs_per_seller_removed = A * n_orders_per_seller_removed**0.5

profits_per_seller_removed = gross_profits_per_seller_removed - it_costs_per_seller_removed
margin_per_seller_removed = profits_per_seller_removed / revenues_per_seller_removed
