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


# Conduct weekly data analysis

pd.to_datetime is a vital preprocessing work to do before plotting data.

'DatetimeIndexResample' helps to reorganize data to show a weekly plot.

``` bash

df['order_approved_at'] = pd.to_datetime(df['order_approved_at'])
df.set_index('order_approved_at').resample('W').plot()

```
