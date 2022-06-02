# Decision Science Project - Olist

# Build a class -> Olist()

One can read data in a neat way with the help of class.

``` bash

# concat different dfs into a dictionary
dfs = {}
for x,y in zip(names,df_list):
    dfs[x] = y

```

Pandas ProfileReport shows almost everything necessary to get a general view of data.

``` bash

profile = ProfileReport(df,title = i)
profile.to_file(f"{i}.html")

```

# Train data -> Order()

Merge several dataframes first to prepare for statistical inference.

``` bash

customers.merge(orders, on='customer_id')\
         .merge(order_items, on='order_id')\
         .merge(sellers, on='seller_id')\
         [['order_id', 'customer_id','customer_zip_code_prefix', 'seller_id', 'seller_zip_code_prefix']]

```

Map your aggregation on specific rows or columns.

``` bash

df.groupby('customer_state').agg({
    'review_score':[np.mean,state_promoter],
    'customer_zip_code_prefix':pd.Series.count
})

```

# Linear model -> statsmodels.formula.api

First, set target and features.

``` bash

# a neat way to define formula
formula = target + ' ~ ' + ' + '.join(products_selected_features)

```

Second, standardize features if necessary.

``` bash

for f in products_selected_features:
    mu = products[f].mean()
    sigma = products[f].std()
    products_standardized[f] = products[f].map(lambda x: (x - mu) / sigma)

```

Third, output model summary to check statistical significance, coeffiency,std err and etc.

``` bash

import statsmodels.formula.api as smf
model = smf.ols(formula = formula, data = products_standardized).fit()
print(model.summary())

```

# pd.DataFrame.resample()

pd.to_datetime is a vital preprocessing work to do before plotting data.

'DatetimeIndexResample' helps to reorganize data to show a weekly plot.

``` bash

df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
df.set_index('order_approved_at').resample('W').plot()

```
