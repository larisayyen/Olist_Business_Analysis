
import os
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport

class Olist:

    def get_data(self):

        #find where csv_files locate
        file_path = 'raw_data/csv'
        file_name = os.listdir(file_path)
        file_name.remove('.gitkeep')

        #standardize file names
        names = []
        for n in file_name:
            if n[:6] == 'olist_':
                n = n.replace('olist_','')
                if n[-4:] == '.csv':
                    n = n.replace('.csv','')
                    if n[-8:] == '_dataset':
                        n = n.replace('_dataset','')
                        names.append(n)

            else:
                n = n.replace('.csv','')
                names.append(n)

        #make a df list
        df_list = []
        for i in file_name:
            df = pd.read_csv(os.path.join(file_path,i))
            df_list.append(df)


        #make a df dict
        dfs = {}
        for x,y in zip(names,df_list):
            dfs[x] = y

        return dfs

    #export pandas profile report
    def explore_data(self,data):
        for i in data:
            df = Olist().get_data()[i]
            profile = ProfileReport(df,title = i)
            profile.to_file(f"raw_data/html/{i}.html")


    #check nunique ratio
    def count_unique(self,data):
        aggs = np.array([[
            data["orders"].order_id.nunique(),
            data["order_reviews"].review_id.nunique(),
            data["sellers"].seller_id.nunique(),
            data["products"].product_id.nunique(),
            data["customers"].customer_id.nunique(),
        ],
        [
            data["orders"].order_id.count(),
            data["order_reviews"].review_id.count(),
            data["sellers"].seller_id.count(),
            data["products"].product_id.count(),
            data["customers"].customer_id.count(),
        ]])

        df = pd.DataFrame(data=np.vstack((aggs, aggs[0,:]/aggs[1,:])).T,
             index=['orders', 'reviews', 'sellers', 'products', 'customers'],
             columns=['nunique', 'count','ratio'])

        df.to_csv('raw_data/count_unique.csv')


# if __name__ == '__main__':

#     # olist_dfs = Olist().get_data()
#     # Olist().explore_data(olist_dfs.keys())
#     # Olist().count_unique(olist_dfs)
