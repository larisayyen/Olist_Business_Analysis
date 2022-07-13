
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Olist_Business_Analysis.data_preparation import Olist

def weekly_payment(data):

    payments = data['order_payments'].copy()
    orders = data['orders'].copy()

    df = payments.groupby('order_id').agg({'payment_value':'sum'})
    df.rename(columns = {'payment_value':'weekly_sum'},inplace = True)

    df_ = orders.merge(df,on='order_id')

    # use Datetime Resample function
    for i in ['order_approved_at','order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']:
        df_[i] = pd.to_datetime(df_[i])

    df_new = df_.set_index('order_approved_at').resample('W')

    # plot the result
    df_new.count()['order_id'].plot(label='order_count')
    df_new.sum()['weekly_sum'].apply(lambda x: x/1000).plot(label='weekly payment')
    df_new.mean()['payment_sum'].plot(label='mean payment')
    plt.legend()
    plt.title('Weekly orders values')

def weekly_statisfaction(data):

    reviews = data['order_reviews'].copy()
    reviews['review_creation_date'] = pd.to_datetime(reviews['review_creation_date'])
    reviews.set_index('review_creation_date',inplace = True)

    reviews.resample('W').agg({'review_score':'mean'}).plot()


def weekly_delay(data):

    orders = data['orders'].copy()
    orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
    orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
    orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

    #aggragate delay
    orders['delay_vs_expected'] = (orders['order_delivered_customer_date'] -\
                                orders['order_estimated_delivery_date']) / np.timedelta64(24, 'h')

    orders['wait_time'] = (orders['order_delivered_customer_date'] -\
                       orders['order_purchase_timestamp']) / np.timedelta64(24, 'h')

    orders["was_late"] = orders['delay_vs_expected'].map(lambda x: x > 0)
    orders["early_or_on_time"] = orders["was_late"].map({True:"late",False:"in_time"})

    delay_analysis_per_week = orders.set_index('order_purchase_timestamp').resample('W').agg({
                                        'delay_vs_expected':np.mean,
                                        'wait_time':np.mean,
                                        'was_late':np.sum,
                                        'order_id':'count'})

    delay_analysis_per_week.columns = ['avg_delay_vs_exp', 'avg_wait_time','nb_of_delays', 'nb_of_orders']

    delay_analysis_per_week['pct_of_lateness'] = delay_analysis_per_week['nb_of_delays'] / delay_analysis_per_week['nb_of_orders']

    lateness_20_and_more = delay_analysis_per_week.query("pct_of_lateness >= 0.20")
    lateness_15_20 = delay_analysis_per_week[delay_analysis_per_week["pct_of_lateness"].between(0.15,0.20)]
    lateness_10_15 = delay_analysis_per_week[delay_analysis_per_week["pct_of_lateness"].between(0.10,0.15)]
    lateness_05_10 = delay_analysis_per_week[delay_analysis_per_week["pct_of_lateness"].between(0.05,0.10)]
    lateness_05_and_less = delay_analysis_per_week[delay_analysis_per_week["pct_of_lateness"].between(0.00,0.05)]

    lateness_brackets = pd.Series([
                                    " ≥ 20%",
                                    "15% - 20%",
                                    "10% - 15%",
                                    "5% - 10%",
                                    " ≤ 5%"
                                ])

    lateness_numbers = pd.Series([
        lateness_20_and_more.shape[0]-1,
        lateness_15_20.shape[0],
        lateness_10_15.shape[0],
        lateness_05_10.shape[0],
        lateness_05_and_less.shape[0]

    ])

    lateness_df = pd.concat([lateness_brackets,lateness_numbers],axis = 1)
    lateness_df.columns = ['brackets', 'nb_weeks']

    lateness_df.to_csv('raw_data/lateness_df.csv')


# if __name__ == '__main__':

#     data = Olist().get_data()
#     weekly_payment(data)
#     weekly_statisfaction(data)
#     weekly_delay(data)
