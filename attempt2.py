import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import folium
import sqlite3
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from IPython.display import HTML

with st.echo(code_location='below'):

    '''
    ## Forbes lists
    Привет! Сейчас мы будем развлекаться со списками Forbes (ну и не только).
    Мы посмотрим на список миллиардеров, на список самых успешных компаний 2022,
    а на некоторых из этих компаний остановимся подробнее. Enjoy :)
    '''

    '''
    Для начала выясним, какие вообще бывают списки Forbes, потому что с чем их только не составляют)
    Для этого воспользуемся библиотекой requests и вытащим с сайта Forbes названия некоторых их списков.
    '''
    r = requests.get('https://www.forbes.ru/ratings')
    soup = BeautifulSoup(r.text)
    heads = []
    for i in range(len(soup.find_all(class_="_3Ew4G"))):
        heads.append(soup.find_all(class_="_3Ew4G")[i].text)
    for i in [2, 3]:
        newurl = 'https://www.forbes.ru/ratings' + '?page=' + str(i)
        r = requests.get(newurl)
        soup = BeautifulSoup(r.text)
        for i in range(len(soup.find_all(class_="_3Ew4G"))):
            heads.append(soup.find_all(class_="_3Ew4G")[i].text)
    q = pd.DataFrame(heads, columns=(['Заголовки списков Forbes']))
    st.write(q)

    """
    Думаю, этого достаточно, чтобы понять, что списков у них дофига ... Но нас сейчас будет
    интересовать один конкретный: Forbes Billionaires.  Вот он:
    """
    df = pd.read_csv('https://raw.githubusercontent.com/raccoon75/project/master/Forbes%20Billionaires.csv')
    df = df.set_index(df.iloc[:, 0])
    df = df.iloc[:, 1:]
    st.write(df)

    """
    Давай посмотрим, какие отрасли и в какой пропорции представлены в списке.
    """

    industries = df["Industry"].unique()
    dictionary1 = {}
    keys1 = []
    size_of_groups = []
    for ind in industries:
        q = df[df['Industry'] == ind]["Name"].count()
        keys1.append(ind)
        dictionary1[ind] = q
        size_of_groups.append(q)
    ind_name = pd.DataFrame(list(dictionary1.items())).rename(
        columns={0: "Industry", 1: "Number of nominants"}).sort_values(by="Number of nominants", ascending=False)

    top10ind = list(ind_name['Number of nominants'])
    rest = []
    for i in top10ind:
        if top10ind[0] / i > 11:
            rest.append(i)
    length = len(top10ind) - len(rest)
    listforchart = top10ind[:length]
    listforchart.append(sum(rest))
    data_names = list(ind_name["Industry"])[:length]
    data_names.append('other')
    totall = sum(listforchart)
    labels = [f"{n} ({v / totall:.1%})" for n, v in zip(data_names, listforchart)]
    colors = [plt.cm.afmhot(i / float(len(listforchart))) for i in range(len(listforchart))]
    fig, ax = plt.subplots()
    ax = plt.pie(listforchart, radius=1.5, colors=colors)
    plt.legend(
        bbox_to_anchor=(-0.4, 0.8, 0.25, 0.25),
        loc='best', labels=labels)
    plt.title(label="Структура индустрий в списке Forbes", fontsize=13, pad=35)
    st.pyplot(fig, figsize = (2,2))

    """
    Интересно, каким странам соответствуют "самые дорогие" строчки Forbes"? Посчитаем среднее богатство всех номинантов списка по странам.
    Для этого воспользуемся чуть более продвинутыми функциями pandas – groupby,agg.
    """
    h = df.groupby('Country').agg({'Networth': 'mean'}).sort_values('Networth', ascending=False)
    st.write(h)

    """
    Как видим, самая крутая тут, похоже, Франция – в среднем у нее самые состоятельные номинанты. А теперь сделаем pivot
    table, чтобы увидеть, какие страны представляют какие отрасли.
    """
    j = df.pivot_table(index='Country',columns='Industry',values='Networth')
    list1 = list(j.columns)
    y = pd.DataFrame(j, columns=(list1))
    y = y.sort_values('Country', ascending=False)
    st.write(y)
    """
    Любопытно, хотя наверное неудивительно, но США – чуть ли не единственная страна, где есть номинатны Форбс во всех отраслях.
    
    """

    """
    А что же Россия?)) Давай посмотрим, в каких отраслях в нашей стране есть номинатны Forbes. Для разнообразия и наглядности 
    сделаем это с помощью графа.
    """
    H = nx.Graph()
    H.add_node('Russia')
    H.add_nodes_from(['Metals,Mining', 'Tech', 'Energy',
                      'Finance', 'Manufacture',
                      'Fashion,Retail', 'Healthcare', 'Logistics', 'Service',
                      'Engineering', 'Telecom', 'Real Estate',
                      'Food,Beverage', 'Automotive'])
    H.add_edges_from([('Russia', 'Metals,Mining'), ('Russia', 'Tech'), ('Russia', 'Energy'),
                      ('Russia', 'Finance'), ('Russia', 'Manufacture'), ('Russia', 'Fashion,Retail'),
                      ('Russia', 'Healthcare'),
                      ('Russia', 'Logistics'), ('Russia', 'Service'), ('Russia', 'Engineering'),
                      ('Russia', 'Telecom'), ('Russia', 'Real Estate'), ('Russia', 'Food,Beverage'),
                      ('Russia', 'Automotive')])
    options = {
        'node_color': 'white',
        'node_size': 1000,
        'edge_color': 'tomato',
        'width': 2,
        'font_size': 11,
        'font_color': 'navy'
    }
    fig, ax = plt.subplots()
    ax = nx.draw(H, with_labels=True, **options)
    st.pyplot(fig)

    """
    Заметим, что лидеры списка Forbes – Илон Маск, Джефф Безос и Бернар Арно, владельцы компаний Tesla, Amazon и 
    LVMH соответственно. Остановимся-ка теперь на этой троице подробнее))
    """

    """
    Для начала посмотрим, в каких странах представлены основные офисы этих компаний (почему б и нет). Для этого воспользуемся 
    библиотекой folium (адреса взяты с сайта craft.co).
    """

    #адреса взяты с сайта craft.co
    teslalocations = pd.DataFrame({'Place': ['USA, Austin', 'NY', 'Fremont', 'Sparks', 'Sydney', 'Beijing', 'Shanghai',
                                             'Grünheide', 'Prüm', 'Hong Kong'],
                                   'Lat': [30.231151900544642, 42.859230930612966, 37.49242895839669, 39.54558357520227,
                                           -33.84125707569156, 39.911431995610506, 30.868123189789596,
                                           52.392630039837556, 50.21428168380385, 22.323362576055423],
                                   'Lon': [-97.61400252290116, -78.84200725019446, -121.94471864885227,
                                           -119.4514283557927, 151.20732388172016, 116.47958816795548,
                                           121.76927248029989, 13.790252130656874, 6.447677609764756,
                                           114.20405467575651]}, columns=(['Place', 'Lat', 'Lon']))

    amazonloc = pd.DataFrame({'Place': ['Seattle', 'Arlington', 'Atlanta', 'Austin', 'Baltimore', 'Bellevue',
                                        'Boardman', 'Boulder', 'Breinigsville', 'Brooklyn'],
                              'Lat': [47.622359451005885, 38.85824152059593, 33.84596260574857, 30.400994528081036,
                                      39.26757930468095, 47.61426626933748, 45.852158497020476, 40.017999262815806,
                                      40.558307516476326, 40.65910423575058, ],
                              'Lon': [-122.3369089875389, -77.05065553931252, -84.37160333744903, -97.71950116214539,
                                      -76.5505568584298,
                                      -122.19966240946134, -119.63017981825233, -105.27551557349578, -75.61376935628246,
                                      -74.00408120045864]})

    lvmh = folium.Map([48.84560187107419, 1.981216172176683], zoom_start=2)
    folium.Marker(location=[48.866328570425864, 2.305572544478632],
                  tooltip='LVMH', icon=folium.Icon(icon='', color='purple')).add_to(lvmh)
    for ind, row in teslalocations.iterrows():
        folium.Marker([row.Lat, row.Lon],
                      tooltip='Tesla', icon=folium.Icon(icon='', color='green')).add_to(lvmh)
    for ind, row in amazonloc.iterrows():
        folium.Marker([row.Lat, row.Lon],
                      tooltip='Amazon', icon=folium.Icon(icon='', color='blue')).add_to(lvmh)

    st.write(lvmh)

    """
    Как обстоят дела у этих трех компаний? Проанализируем поведение их акций. Ниже представлены графики изменения цен акиций
    Амазона, LMVH и Теслы соответственно.
    """
    amazondf = pd.read_csv('https://raw.githubusercontent.com/raccoon75/project/master/AMZN.csv')
    amazondf = amazondf.dropna().assign(Date=lambda x: pd.to_datetime(x['Date'])).set_index('Date')
    fig, ax = plt.subplots()
    amazondf.iloc[:].plot(y='Adj Close', ax=ax, label='Adj Close price', color='darkorange')
    amazondf.iloc[:].rolling(182, center=True).mean().plot(y='Adj Close', ax=ax, label='Half-year average',
                                                           color='darkred')
    plt.ylabel('Stock price, $', fontsize=11)
    plt.xlabel('Date', fontsize=11)
    plt.title('Динамика акций Amazon, 2010-2022', fontsize=12)
    plt.legend(fontsize=11)
    st.pyplot(fig,figsize = (2,2))

    lmvhdf = pd.read_csv('https://raw.githubusercontent.com/raccoon75/project/master/LVMUY.csv')
    lmvhdf = lmvhdf.dropna().assign(Date=lambda x: pd.to_datetime(x['Date'])).set_index('Date')
    fig, ax = plt.subplots()
    lmvhdf.iloc[:].plot(y='Adj Close', ax=ax, label='Adj Close price', color='dodgerblue')
    lmvhdf.iloc[:].rolling(182, center=True).mean().plot(y='Adj Close', ax=ax, label='Half-year average',
                                                         color='fuchsia')
    plt.ylabel('Stock price, $', fontsize=11)
    plt.xlabel('Date', fontsize=11)
    plt.title('Динамика акций LVMH, 2010-2022', fontsize=12)
    plt.legend(fontsize=11)
    st.pyplot(fig,figsize = (2,2))

    """
    Данные про акции Теслы для разнообразия обработаем посредством SQL ))
    """
    conn = sqlite3.connect("data.sqlite")
    c = conn.cursor()
    teslasqldf = pd.read_csv('https://raw.githubusercontent.com/oubielamir/Tesla-Stock-Analysis/master/TSLA.csv')
    teslasqldf.to_sql('tesla', conn)
    teslasqldf = teslasqldf.rename(columns={'Adj Close': 'AdjClose'})
    tesladf = pd.read_sql(
        """
    SELECT Date, Close FROM tesla
    """,
        conn,
    )
    tesladf = tesladf.assign(Date=lambda x: pd.to_datetime(x['Date'])).set_index('Date')

    fig, ax = plt.subplots()
    tesladf.iloc[:].plot(y='Close', ax=ax, label='Close price', color='slategray')
    tesladf.iloc[:].rolling(182, center=True).mean().plot(y='Close', ax=ax, label='Half-year average',
                                                          color='springgreen')
    plt.ylabel('Stock price, $', fontsize=11)
    plt.xlabel('Date', fontsize=11)
    plt.title('Динамика акций Tesla, 2010-2022', fontsize=12)
    plt.legend(fontsize=11)
    st.pyplot(fig, figsize = (2,2))

    """
    В 2021-2022 году, как видим, у всех трех компаний цены акций улетели в небеса. Теперь используем numpy, чтобы посчитать доходность и волатильность акций всех трех компаний, и сравним их друг с другом.
    """
    a = np.array(tesladf['Close'])
    x = np.mean(a)
    y = np.var(a)
    st.write('Тесла, доходность: ',x,'; ','Тесла, волатильность: ',np.sqrt(y) )

    b = np.array(amazondf['Adj Close'])
    x1 = np.mean(b)
    y1 = np.var(b)
    st.write('Амазон, доходность: ', x1, '; ', 'Амазон, волатильность: ', np.sqrt(y1))

    c1 = np.array(lmvhdf['Adj Close'])
    x2 = np.mean(c1)
    y2 = np.var(c1)
    st.write('LMVH, доходность: ', x2, '; ', 'LMVH, волатильность: ', np.sqrt(y2))

    diag1 = pd.DataFrame({'Name': ['Tesla', 'Tesla', 'Amazon', 'Amazon', 'LVMH', 'LVMH'],
                          'Data': [float(x), float(np.sqrt(y)), float(x1), float(np.sqrt(y1)), float(x2),
                                   float(np.sqrt(y2))],
                          'Type': ['Returns', 'Standard deviation', 'Returns', 'Standard deviation', 'Returns',
                                   'Standard deviation']})

    compare = sns.catplot(
        data=diag1, kind="bar",
        x="Name", y="Data", hue="Type",
        ci="sd", palette="dark", alpha=.6, height=6
    )
    compare.despine(left=True)
    compare.set_axis_labels("", "Company")
    compare.legend.set_title("")
    st.pyplot(compare)

    """
    И, как можно наблюдать, самый рискованный, но при этом и самый доходный актив из трех – акции Теслы.
    """

    """
    Кстати про Теслу. Почему бы не выяснить, какие модели продаются сейчас в Москве?) Используем сайт moscowteslaclub и 
    библиотеку selenium, чтобы достать список и характеристики моделей в продаже (ps код для вебскреппинга в отдельном файле,
    чтобы проект не грузился миллион лет, а здесь просто данные из готового csv-файла. Поэтому цены могут отличаться).
    """
    finaltesla = pd.read_csv('https://raw.githubusercontent.com/raccoon75/project/master/tesla_models.csv').drop('Unnamed: 0', axis=1)
    word = 'See model'
    e = pd.Series(['https://moscowteslaclub.ru/cars/tesla-model-3-long-range-awd_32019/',
                   'https://moscowteslaclub.ru/cars/tesla-model-3-long-range-awd_26262/',
                   'https://moscowteslaclub.ru/cars/tesla-model-3-long-range-awd_30851/',
                   'https://moscowteslaclub.ru/cars/tesla-model-3-performance-2021_26923/',
                   'https://moscowteslaclub.ru/cars/tesla-model-3-standard-plus-2021_26924/',
                   'https://moscowteslaclub.ru/cars/tesla-model-s-long-range_39010/',
                   'https://moscowteslaclub.ru/cars/tesla-model-s-long-range_621/',
                   'https://moscowteslaclub.ru/cars/tesla-model-s-long-range_32875/',
                   'https://moscowteslaclub.ru/cars/tesla-model-s-plaid-2021_28871/',
                   'https://moscowteslaclub.ru/cars/tesla-model-s-plaid-2021_28869/',
                   'https://moscowteslaclub.ru/cars/tesla-model-s-plaid-2021_28868/',
                   'https://moscowteslaclub.ru/cars/tesla-model-x-90d-2016_37741/',
                   'https://moscowteslaclub.ru/cars/tesla-model-x-long-range-2021_26926/',
                   'https://moscowteslaclub.ru/cars/tesla-model-x-p100d-2018_33158/',
                   'https://moscowteslaclub.ru/cars/tesla-model-x-plaid-2022_33345 /',
                   'https://moscowteslaclub.ru/cars/tesla-model-x-plaid_18255/',
                   'https://moscowteslaclub.ru/cars/tesla-model-x-plaid-2021_26925/',
                   'https://moscowteslaclub.ru/cars/tesla-model-y-long-range-awd_11284/',
                   'https://moscowteslaclub.ru/cars/tesla-model-y-long-range-awd_32872/',
                   'https://moscowteslaclub.ru/cars/tesla-model-y-long-range-awd_7077/',
                   'https://moscowteslaclub.ru/cars/tesla-model-3-performance-2021_26923/',
                   'https://moscowteslaclub.ru/cars/tesla-roadster_1956/',
                   'https://moscowteslaclub.ru/cars/tesla-semi_1955/'])
    finaltesla['Link'] = e

    finaltesla["Hyperlink"] = '<a href="' + finaltesla.Link + '">' + word + '</a>'
    finaltesla = finaltesla.drop('Link', axis = 1)
    finaltesla = HTML(finaltesla.to_html(escape=False))
    st.write(finaltesla)
    
    """
    На этом оставим Теслу в покое. И вернемся к Forbes). Нас интересует еще один список – список крупнейших мировых компаний 2022.
    Вот он:
    """
    companies = pd.read_csv('https://raw.githubusercontent.com/raccoon75/project/master/forbes%20global%202022(2000%20companies)%20-%20companies.csv')
    companies = companies.set_index(companies.iloc[:, 0])
    companies = companies.iloc[:, 1:]
    companies = companies.rename(columns={'global company': 'Company', 'country': 'Country', 'sales': 'Sales', 'profit': 'Profit'})
    st.write(companies)

    """
    Здесь для каждой компании указаны их продажи, выручка, стоимость активов и рыночная стоимость компании. Можем посмотреть,
    как все эти показатели коррелируют между собой.
    """
    a1 = np.array(companies.iloc[:, 2])
    a2 = np.array(companies.iloc[:, 3])
    a3 = np.array(companies.iloc[:, 4])
    a4 = np.array(companies.iloc[:, 5])

    com = pd.DataFrame({'Sales': a1, 'Profits': a2, 'Assets': a3, 'Mv': a4})
    com = (com.assign(Sales=lambda x: x['Sales'].str[:-2]).assign(
        Sales=lambda x: x['Sales'].str[1:].str.replace(",", "").astype(float))
           .assign(Profits=lambda x: x['Profits'].str[:-2]).assign(
        Profits=lambda x: x['Profits'].str[1:].str.replace(",", "").astype(float))
           .assign(Assets=lambda x: x['Assets'].str[:-2]).assign(
        Assets=lambda x: x['Assets'].str[1:].str.replace(",", "").astype(float))
           .assign(Mv=lambda x: x['Mv'].str[:-2]).assign(
        Mv=lambda x: x['Mv'].str[1:].str.replace(",", "").astype(float)))

    corr = com.corr()
    st.write(corr)

    """
    Заметим, что самая сильная здесь корреляция – между рыночной стоимостью и продажами. Используем линейную регрессию, чтобы
    попробовать смоделировать продажи, зная рыночную стоимость.
    """
    m = LinearRegression()
    m.fit(com[["Mv"]], com["Sales"])
    st.write("Коэффициенты модели: ", m.coef_[0], ',', m.intercept_)

    """
    Посмотрим на графике:
    """

    fig, ax = plt.subplots()
    plt.scatter(x=com["Mv"], y=com["Sales"], label='Sales Data', color='#B8860B')
    x = pd.DataFrame(dict(Mv=np.linspace(0, 2500)))
    plt.plot(x["Mv"], m.predict(x), label='Prediction', color="seagreen", lw=3)
    plt.legend()
    st.pyplot(fig,figsize=(2, 2))

    """
    Попробуем теперь регрессию от всех переменных (mv, profit, assets ), чтобы понять, можно ли с их помощью моделировать продажи:
    """
    multm = LinearRegression()
    multm.fit(com.drop(columns=["Sales"]), com["Sales"])
    st.write("Коэффициенты модели: ", multm.coef_)
    """
    И посмотрим, как теперь модель соотносится с данными:
    """
    fig, ax = plt.subplots()
    plt.scatter(x=com["Mv"], y=com["Sales"], label='Sales data', color='#B8860B')
    plt.plot(com["Mv"], multm.predict(com.drop(columns=["Sales"])), 'o', label='Prediction', color='seagreen', alpha=0.15)
    plt.legend()
    st.pyplot(fig, figsize = (2,2))

    """ И наконец, сделаем то же самое методом ближайших соседей"""
    kn = KNeighborsRegressor()
    kn.fit(com.drop(columns=["Sales"]), com["Sales"])

    fig, ax = plt.subplots()
    plt.scatter(x=com["Mv"], y=com["Sales"], label='Sales data', color='#B8860B')
    plt.plot(
        com["Mv"],
        kn.predict(com.drop(columns=["Sales"])),
        "o",label='Prediction',
        color="seagreen",
        alpha=0.2)
    plt.legend()
    st.pyplot(fig,figsize = (2,2))

    """
    Осталось только понять, какая из моделек работает точнее. Посмотрим на их среднеквадратичные ошибки:
    """

    st.write('MSE для метода ближайших соседей: ', ((com["Sales"] - kn.predict(com.drop(columns=["Sales"]))) ** 2).mean())
    st.write('MSE для линейной регрессии от нескольких переменных: ',
          ((com["Sales"] - multm.predict(com.drop(columns=["Sales"]))) ** 2).mean())
    st.write('MSE для линейной регрессии от одной переменной: ', ((com["Sales"] - m.predict(com[["Mv"]])) ** 2).mean())

    """
    Что ж, похоже лучше всех справляется метод ближайших соседей. И то выдает большую MSE, так что
    можем сделать вывод, что предсказывать продажи на основе рыночной стоимости компании – плохая идея:)
    """

    """
    ### На этом все, спасибо!
    """

    """
    Чтобы было проще проверять, распишу, что было в проекте:\n
    - Работа с продвинутыми функциями pandas (pivot tables, agg, groupby) в начале\n
    - Работа с api (собирали заголовки списков форбс)\n
    - Визуализации (ни на что не намекаю, но код вроде бы вполне не тривиальный)\n
    - Графы с помощью библиотеки networkx (структура индустрий России в Форбс)\n
    - Работа с геоданными (офисы компаний на карте)\n
    - sql (обрабатывали данные про акции Теслы)\n
    - Numpy, чтобы считать доходности, дисперсии и волатильности акций. Еще использовался, чтобы форматировать строки в таблице с крупнейшими компаниями по версии Форбс\n
    - Веб скреппинг с selenium (доставали модели Теслы в продаже в мск. Файл с кодом лежит в архиве отдельно, seleniumstuff. А тут, чтобы быстрее грузилось, просто загружаю готовый датафрейм-результат)\n
    - Машинное обучение (регрессии для моделирования продаж)\n
    - streamlit\n
    - Из технологий, необсужденных на курсе – форматирование строк и использование HTML из IPython display,чтобы сделать гиперссылки в табличке с моделями Теслы.\n
    - Можешь поверить на слово, что строчек кода здесь +_ 160 (еще не считая файла seleniumstuff). Но можешь, конечно, пересчитать :)
    """
