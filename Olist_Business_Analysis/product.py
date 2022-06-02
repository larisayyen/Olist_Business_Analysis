
import pandas as pd
import numpy as np
from Olist_Business_Analysis.data_preparation import Olist
from Olist_Business_Analysis.orders_analysis import Order


class Product:
    def __init__(self):
        # Import data only once
        olist = Olist()
        self.data = olist.get_data()
        self.order = Order()

    def get_product_features(self):

        products = self.data['products']

        # (optional) convert name to English
        en_category = self.data['product_category_name_translation']
        df = products.merge(en_category, on='product_category_name')
        df.drop(['product_category_name'], axis=1, inplace=True)
        df.rename(columns={
            'product_category_name_english': 'category',
            'product_name_lenght': 'product_name_length',
            'product_description_lenght': 'product_description_length'
        },
                  inplace=True)

        return df

    def get_price(self):

        order_items = self.data['order_items']
        # There are many different order_items per product_id, each with different prices. Take the mean of the various prices
        return order_items[['product_id',
                            'price']].groupby('product_id').mean()

    def get_wait_time(self):

        orders_wait_time = self.order.wait_time()
        orders_products = self.data['order_items'][['order_id', 'product_id']].drop_duplicates()
        orders_products_with_time = orders_products.merge(orders_wait_time, on='order_id')

        return orders_products_with_time.groupby('product_id',as_index=False).agg({'wait_time': 'mean'})

    def get_review_score(self):

        orders_reviews = self.order.review_score()
        orders_products = self.data['order_items'][['order_id','product_id']].drop_duplicates()

        df = orders_products.merge(orders_reviews, on='order_id')

        result = df.groupby('product_id', as_index=False).agg({
            'dim_is_one_star':
            'mean',
            'dim_is_five_star':
            'mean',
            'review_score':
            'mean',
        })

        result.columns = [
            'product_id', 'share_of_one_stars', 'share_of_five_stars',
            'review_score'
        ]

        return result

    def get_quantity(self):

        order_items = self.data['order_items']

        n_orders = order_items.groupby('product_id')['order_id'].nunique().reset_index()
        n_orders.columns = ['product_id', 'n_orders']

        quantity = order_items.groupby('product_id',as_index=False).agg({'order_id': 'count'})

        quantity.columns = ['product_id', 'quantity']

        return n_orders.merge(quantity, on='product_id')

    def get_sales(self):

        return self.data['order_items'][['product_id', 'price']]\
            .groupby('product_id')\
            .sum()\
            .rename(columns={'price': 'sales'})

    def get_training_data(self):

        training_set =\
            self.get_product_features()\
                .merge(
                self.get_wait_time(), on='product_id'
               ).merge(
                self.get_price(), on='product_id'
               ).merge(
                self.get_review_score(), on='product_id'
               ).merge(
                self.get_quantity(), on='product_id'
               ).merge(
                self.get_sales(), on='product_id'
               )

        return training_set

    def get_product_cat(self, agg="mean"):

        products = self.get_training_data()

        columns = list(products.select_dtypes(exclude=['object']).columns)
        agg_params = dict(zip(columns, [agg] * len(columns)))
        agg_params['quantity'] = 'sum'

        product_cat = products.groupby("category").agg(agg_params)

        return product_cat
