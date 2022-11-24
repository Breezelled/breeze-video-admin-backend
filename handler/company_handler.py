import pandas as pd


def clean_company(df):
    df = df.assign(company=df['company'].str.split('|')).explode('company')
    company_dict = {"20th Century Fox": "20th Century Fox",
                    "Buena Vista": "Disney",
                    "Disney": "Disney",
                    "Warner Bros": "Warner Bros.",
                    "Sony": "Sony",
                    "Columbia": "Columbia",
                    "Paramount": "Paramount"}
    print(type(list(company_dict.values())))
    company_list = list(company_dict.values())
    print(company_list)
    for k, v in company_dict.items():
        df.loc[df['company'].str.contains(k), 'company'] = v
    df = df[df['company'].isin(company_list)]

    # for v in data:
    #     v['rating'].sort()
    # df = df.groupby(['company']).mean().round(2).reset_index()
    # df['cnt'] = 1
    # data = df.groupby(['company', 'rating'])['cnt'].size()
    # data = data.reset_index()
    # data['rating'] = dfr['rating']

    # print(dfr)
    # print(data)

    return df
