
import numpy as np
import pandas as pd

from Olist_Business_Analysis.product import Product
from Olist_Business_Analysis.seller_analysis import Seller

products = Product().get_training_data().copy()
sellers = Seller().get_training_data().copy()

# compute revenues
olist_revenues_from_monthly_subscriptions = sellers.months_on_olist.sum()*80

olist_revenues = products.revenues.sum() + olist_revenues_from_monthly_subscriptions
olist_gross_profits = olist_revenues - products.cost_of_reviews.sum()

sorted_products = products.sort_values(by='profits')[['profits', 'n_orders', 'revenues']].reset_index()

# Olist's net_profits for various seller cut-offs

# Analysis excluding IT costs
revenues_per_product_removed = olist_revenues - np.cumsum(sorted_products.revenues)
gross_profits_per_product_removed = olist_gross_profits - np.cumsum(sorted_products.profits)

# Add the IT costs of Olist's platform
olist_it_costs_all_orders = 500000
A = olist_it_costs_all_orders / (sellers['n_orders'].sum()**0.5)

n_orders_per_product_removed = sorted_products.n_orders.sum() - np.cumsum(sorted_products.n_orders)
it_costs_per_product_removed = A * n_orders_per_product_removed**0.5

profits_per_product_removed = gross_profits_per_product_removed - it_costs_per_product_removed
margin_per_product_removed = profits_per_product_removed / revenues_per_product_removed
