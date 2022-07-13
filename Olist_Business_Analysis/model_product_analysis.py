
import pandas as pd
import statsmodels.formula.api as smf
from Olist_Business_Analysis.product import Product

products = Product().get_training_data()

# # Which feature of product effects review score?

# calculate product volume
products['product_volume_cm3'] = products["product_length_cm"] * products["product_height_cm"] * products["product_width_cm"]

# select target and features
target = 'review_score'
products_selected_features = [
    'product_photos_qty',
    'product_volume_cm3',
    'wait_time',
    'price',
    'quantity',
]

# standardize features
products_standardized = products.copy()
for f in products_selected_features:
    mu = products[f].mean()
    sigma = products[f].std()
    products_standardized[f] = products[f].map(lambda x: (x - mu) / sigma)

# linear model
formula = target + ' ~ ' + ' + '.join(products_selected_features)
model = smf.ols(formula = formula, data = products_standardized).fit()
# print(model.summary())
# model.params[1:].sort_values().plot(kind='barh')

# # What are the numerical columns of the products dataset ?
numerical_columns = products.select_dtypes(exclude = ["object"]).columns

# # What kind of products effect wait time?
products_std = products.copy()
for f in numerical_columns:
    mu = products[f].mean()
    sigma = products[f].std()
    products_std[f] = products[f].map(lambda x: (x - mu) / sigma)

model = smf.ols(formula='review_score ~ C(category) + wait_time', data=products).fit()

from olist.utils import return_significative_coef
return_significative_coef(model)[1:6]


# # What are the top products?
products_per_category = products.groupby("category")['product_id'].\
                                        count().\
                                        sort_values(ascending = False)
products_per_category = pd.DataFrame(products_per_category).reset_index()
products_per_category.columns = ["category", "nb_of_unique_products"]
top_20_products = products_per_category.head(20)


product_cat = Product().get_product_cat(agg="mean")

top_5 = round(product_cat.reset_index()[['category','review_score']]\
    .sort_values(by='review_score', ascending = False)\
    .head(5),2)

last_5 = round(product_cat.reset_index()[['category','review_score']]\
    .sort_values(by='review_score', ascending = False)\
    .tail(5),2)

# product_cat['sales'] = product_cat['price'] * product_cat['quantity']

# interesting_features = set(product_cat.columns) - set(["review_score",
#                                                        "share_of_five_stars",
#                                                        "share_of_one_stars",
#                                                        "product_description_length",
#                                                        "product_height_cm",
#                                                        "product_length_cm",
#                                                        "product_width_cm"
#                                                       ])

# round(product_cat[interesting_features].describe(),2)

# q1 = product_cat["product_volume_cm3"].describe()["25%"]
# q3 = product_cat["product_volume_cm3"].describe()["75%"]
# iqr = q3 - q1
# lower_bound = q1 - 1.5 * iqr
# upper_bound = q3 + 1.5 * iqr

# import plotly.express as px

# fig = px.scatter(
#     product_cat[product_cat['product_volume_cm3'] > upper_bound].reset_index(),
#     x="wait_time",
#     y="product_volume_cm3",
#     size="sales",
#     hover_name="category",
#     color="review_score",
#     size_max=40,
# )
# fig.show()
