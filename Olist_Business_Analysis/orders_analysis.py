
import numpy as np
import pandas as pd
from Olist_Business_Analysis.data_preparation import Olist
from Olist_Business_Analysis.utils import haversine_distance


class Order:

    def __init__(self):
        self.data = Olist().get_data()


    def wait_time(self,is_delivered = True):

        #get delivered orders
        orders = self.data['orders'].copy()
        orders=orders.query("order_status == 'delivered'")

        #aggregate wait time
        orders.loc[:,'order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders.loc[:,'order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        orders.loc[:,'order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

        orders.loc[:,'wait_time'] = (orders.order_delivered_customer_date - orders.order_purchase_timestamp) / np.timedelta64(24,'h')
        orders.loc[:,'expected_wait_time'] = (orders.order_estimated_delivery_date - orders.order_purchase_timestamp) / np.timedelta64(24,'h')
        orders.loc[:,'delay_vs_expected'] = (orders.order_delivered_customer_date - orders.order_estimated_delivery_date) / np.timedelta64(24,'h')
        orders.loc[:,'delay_vs_expected'] = orders.delay_vs_expected.apply(lambda x: x if x>0 else 0)

        orders = orders[['order_id','order_status','wait_time', 'expected_wait_time', 'delay_vs_expected']]

        return orders


    def review_score(self):

        #get review df
        reviews = self.data['order_reviews'].copy()

        #aggregate score
        reviews['dim_is_five_star'] = reviews.review_score
        reviews['dim_is_one_star'] = reviews.review_score
        reviews['dim_is_five_star'] = reviews['dim_is_five_star'].apply(lambda x: 1 if x == 5 else 0)
        reviews['dim_is_one_star'] = reviews['dim_is_one_star'].apply(lambda x: 1 if x == 1 else 0)

        reviews = reviews[['order_id', 'review_score', 'dim_is_five_star', 'dim_is_one_star']]

        return reviews

    def product_count(self):

        #get product df
        product = self.data['order_items'].copy()

        #aggregate product numbers
        product = product.groupby('order_id').agg({'product_id':'count'}).rename(columns = {'product_id':'product_count'})
        product = product.reset_index()

        return product

    def seller_count(self):

        #get seller df
        seller = self.data['order_items'].copy()

        #aggregate seller count
        seller = seller.groupby('order_id').agg({'seller_id':'count'}).rename(columns={'seller_id':'seller_count'})
        seller = seller.reset_index()

        return seller


    def price_freight(self):

        #get df
        pf = self.data['order_items'].copy()

        #aggregate prrice sum and freight sum
        pf = pf.groupby('order_id').agg({'price':'sum','freight_value':'sum'})
        pf = pf.reset_index()

        return pf

    def distance(self):

        #get dfs
        data = self.data
        orders = self.data['orders'].copy()
        order_items = self.data['order_items'].copy()
        sellers = self.data['sellers'].copy()
        customers = self.data['customers'].copy()
        geo = self.data['geolocation'].copy()

        #groupby zip_code_prefix
        geo = geo.groupby('geolocation_zip_code_prefix',as_index=False).first()

        #seller df
        sellers_mask_columns = ['seller_id', 'seller_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']
        sellers_geo = sellers.merge(geo,how='left',left_on='seller_zip_code_prefix',right_on='geolocation_zip_code_prefix')[sellers_mask_columns]

        #customer df
        customers_mask_columns = ['customer_id', 'customer_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']
        customers_geo = customers.merge(geo,how='left',left_on='customer_zip_code_prefix',right_on='geolocation_zip_code_prefix')[customers_mask_columns]

        #merge dfs
        customers_sellers = customers.merge(orders, on='customer_id')\
                            .merge(order_items, on='order_id')\
                            .merge(sellers, on='seller_id')\
                            [['order_id', 'customer_id','customer_zip_code_prefix', 'seller_id', 'seller_zip_code_prefix']]

        matching_geo = customers_sellers.merge(sellers_geo,on='seller_id')\
                       .merge(customers_geo,on='customer_id',suffixes=('_seller','_customer'))

        #aggregate distance between seller and customer
        matching_geo = matching_geo.dropna()

        matching_geo.loc[:,'distance_seller_customer'] =matching_geo.apply(lambda row:haversine_distance(row['geolocation_lng_seller'],\
                                                                        row['geolocation_lat_seller'],\
                                                                        row['geolocation_lng_customer'],\
                                                                        row['geolocation_lat_customer']),axis=1)

        #groupby order_id
        distance =matching_geo.groupby('order_id',as_index=False).agg({'distance_seller_customer':'mean'})

        return distance


    def data_trained(self,is_delivered = True):

        orders=self.wait_time()
        reviews=self.review_score()
        df=self.product_count()
        sellers=self.seller_count()
        pay=self.price_freight()
        distance=self.distance()

        dfs=[reviews,df,sellers,pay,distance]
        for i in range(len(dfs)):
            df_data = orders.merge(dfs[i],on='order_id')
            orders = df_data

        df_data = df_data.dropna()

        return df_data

        # df_data.to_csv('raw_data/trained_df.csv')


# if __name__ == '__main__':

#     Order().data_trained(is_delivered=True)
