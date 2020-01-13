#!-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader.data as web
import datetime as dt

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split as SetSplit
from sklearn.model_selection  import cross_val_score as CVS
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from sklearn.tree import export_graphviz as export_graph
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.utils import shuffle

def getStocks(stocks, tweet_dir, stock_dir): # 根据推特文件获取对应的股票数据
    if not (os.path.exists(stock_dir)): # 检测股票文件夹是否存在，若不存在则创建文件夹。
        os.mkdir(stock_dir)
    for item in stocks: # 遍历每一支股票
        file_path = tweet_dir + os.sep * 2 + 'Scored' + os.sep *2 + item + '_scored.xlsx' # 股票文件路径
        df_scored = pd.read_excel(file_path)
        start = df_scored.datetime[0].to_pydatetime() # 获取推特情感得分文件开始时间
        end = df_scored.datetime[df_scored.datetime.size - 1] # 获取推特情感得分文件结束时间
        df_scored = df_scored.set_index(df_scored.datetime) # 将时间作为索引
        df_scored.index.name = 'datetime' # 将索引名更改为datetime
        df_scored = df_scored.drop(['datetime'], axis=1) # 去掉datetime列
        df_stock = web.DataReader(item, 'yahoo', start, end) # 使用Pandas从Yahoo!财经下载股票文件
        df_stock['Pct_change'] = (df_stock['Close'] - df_stock['Open']) / df_stock['Open'] * 100 # 计算每一天的涨跌幅
        df_stock.index.name = 'datetime' # 将下载的股票数据索引名设置为datetime，以便与情感得分文件索引列匹配。
        df = pd.concat([df_scored, df_stock], axis=1) # 合并情感得分数据和股票数据
        columnList = ('Favs', 'RTs', 'Followers', 'Following', 'Is a RT',  
            'compound', 'neg', 'neu', 'pos', 'Compound_multiplied', 'Compound_multiplied_scaled')
        for col in columnList: # 对columnList中含有缺失值的列使用均值进行填充
            df[col].fillna(df[col].mean(), inplace=True)

        df[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'Pct_change']] = df[['High', 'Low',  # 将股票数据中的缺失值采用插值法填充
        'Open', 'Close', 'Volume', 'Adj Close', 'Pct_change']].interpolate(method='linear', limit_direction='forward', axis=0)
        df['Predicted_pct_change'] = df.Pct_change.shift(-1) # 生成预测涨跌幅（即下一天的涨跌幅）
        df['Buy/Sell'] = df.Predicted_pct_change.apply(lambda x: 1 if x >= 0 else - 1) # 根据预测涨跌幅生成Buy/Sell数据
        df.drop(df[np.isnan(df['Buy/Sell'])].index, inplace=True) # 去掉Buy/Sell列含有缺失值的行
        export_path = stock_dir + os.sep * 2 + item + '_concat.xlsx' # 生成情感得分和股票数据的合并文件名
        df.to_excel(export_path) # 导出为Excel文件

def sentimentScore(Tweet):  # 定义情感得分函数，使用Vader对句子进行打分。
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for sentence in Tweet:
        vs = analyzer.polarity_scores(sentence)
        results.append(vs)
    return results # 返回得分字典列表，包括pos、neg、neu、compound的键值。

def scoreTweet(stocks, tweet_dir): # 推特文件情感打分函数，生成对应的情感得分文件。
    for item in stocks:
        file_dir = tweet_dir
        filePath = file_dir + os.sep * 2 + item +'.xlsx'  # 定义打开文件路径
        df = pd.read_excel(filePath, 'Stream')  # 读取excel表格
        datetimeToString = lambda x: dt.datetime.strftime(x, '%Y-%m-%d') if type(x) == dt.datetime else x
        df.Date = df.Date.apply(datetimeToString)
        df = df[(df.Date >= '2016-04-01') & (df.Date <= '2016-06-15')]
        tweets = df['Tweet content']  # 读取推特内容
        scores = pd.DataFrame(sentimentScore(tweets)) # 根据推特内容给出情感得分，生成一个新的DataFrame。
        df = pd.concat([df, scores], axis=1)  # 按列合并
        df.Date = pd.to_datetime(df.Date) # 将string转换为datetime
        df = df.set_index(df.Date)  # 将时间作为索引
        df.drop(['Date'], axis=1)

        value_li = ['Hour', 'Tweet content', 'Favs', 'RTs',
                    'Followers', 'Following', 'Is a RT',
                    'Hashtags', 'Symbols', 'compound','neg', 'neu', 'pos'] # 定义选取的标签列表
        df = pd.DataFrame(df, columns=value_li) # 根据标签列表选取指定的列
        df = df[df.loc[:, 'compound'] != 0] # 选取compound得分不为0的行
        df = df.dropna(subset=['Followers']) # 去掉Followers为空的行
        df.eval('Compound_multiplied = compound * Followers', inplace=True)  # 按照公式计算数据并生成新列

        x = df[['Compound_multiplied']].values.astype('float')
        scaler = StandardScaler().fit(x) # 对Compound_multiplied值进行归一化
        scaled_data = scaler.transform(x)
        df['Compound_multiplied_scaled'] = scaled_data

        grouped = df.groupby(df.index) # 按照日期聚合
        df = grouped.mean()
        df.index.name = 'datetime'  # 修改index名称

        target_dir = file_dir + os.sep * 2 + 'Scored' # 定义得分文件所在文件夹路径
        if not (os.path.exists(target_dir)): # 路径不存在则创建新文件夹
            os.mkdir(target_dir)
        target_path = target_dir + os.sep * 2 + item + '_scored.xlsx' # 定义输出路径
        df.to_excel(target_path)  # 将DataFrame输出为Excel文件

def prediction(stocks, stock_dir, summary_dir, prediction_dir, picture_dir, days): #运用八种机器学习分类器对Buy/Sell值进行预测，生成预测数据准确率的表格和图片。
    markers = ('D', '^', 'v', 'o', 's', 'x', '.', '|') # 定义plt.plot中点的形状参数
    stat = {} # 定义一个统计字典。之后会将股票名作为key，八中预测器的准确率列表作为值存储在字典中。
    if (os.path.exists(prediction_dir) == False):
        os.makedirs(prediction_dir)
    for (n, item) in enumerate(stocks): # 对每一支股票进行遍历，同时n为遍历的索引（从0开始）。
        df = pd.read_excel(stock_dir + os.sep * 2 + item + '_concat.xlsx') # 读取情感得分和股票数据的合并文件
        df.drop(df[np.isnan(df['Predicted_pct_change'])].index, inplace=True) # 去掉预测涨跌幅为空的行
        data_train = df.iloc[:-days] # 选取除去最后days天的行作为训练集
        data_test = df.iloc[-days:] # 选取最后days天的行作为测试集
        data_train.reset_index(inplace=True) # 重新建立索引
        data_test.reset_index(inplace=True)
        feature = ['Favs', 'Followers', 'Is a RT', 'RTs', 'pos', 'neu', 'neg', 'Compound_multiplied_scaled'] # 选取的所有特征
        x_train = np.array(data_train[feature])
        y_train = np.array(data_train['Buy/Sell'])
        x_test = np.array(data_test[feature])
        y_test = np.array(data_test['Buy/Sell'])        
        score = {}
        stat[item] = []
        knn = KNN(bestKNN(item, x_train, y_train, 1, 32, graph=True, picture_dir=picture_dir)) # 使用bestKNN函数确定最佳k_neighbor参数
        lr = LR(solver='liblinear') # 逻辑回归
        nb = MNB() # 朴素贝叶斯
        svc = SVC(kernel='linear') # 支持向量机
        dt = DTC(criterion='gini', splitter='random', random_state=None) # 决策树
        rf = RFC(n_estimators=10) # 随机森林
        mlpc = MLPC(activation='relu', solver='sgd',hidden_layer_sizes= (50,50),alpha=0.0001, shuffle=True, max_iter=300) # 多层感知机
        gbdt = GBDT(n_estimators=3, learning_rate=0.1, max_depth=3) # 梯度下降决策树

        algorithms = (('KNN',knn), ('LogReg',lr), ('NaiveBayes',nb), ('SVM',svc),  
        ('DecisionTree',dt),('RandomForest',rf), ('Perceptron', mlpc), ('GBDT', gbdt))
        for (algorithm, func) in algorithms:
            if algorithm == 'NaiveBayes': # 如果是朴素贝叶斯算法，还需要对训练值进行归一化。
                scaler = MinMaxScaler()
                x_train = scaler.fit_transform(x_train)
            score[algorithm] = CVS(func, x_train, y_train, cv=10, scoring='accuracy').mean() # 10折交叉验证，得到分类器对应的预测准确率均值。
            func.fit(x_train, y_train) # 使用训练集对分类器进行训练
            data_test = data_test.copy() 
            data_test[algorithm + '_prediction'] = func.predict(x_test) # 对测试集进行预测，生成预测的Buy/Sell数据。
            stat[item].append(score[algorithm]) 

        print('\nStock ' + item +' Prediction')
        for key, value in score.items(): # 打印每一种分类器的预测准确率
            print('Algorithm -> %-12s Accuracy -> %.2f%%' % (key, value * 100))
        
        data_test.to_excel(prediction_dir + os.sep * 2 + item + '_prediction.xlsx')
    labels = tuple([item[0] for item in algorithms]) # labels存储了先后遍历到的分类器名称
    accGraph(stat, labels, markers, picture_dir) # 生成预测准确率的图形
    accExcel(stat, labels, summary_dir, rank=10) # 生成预测准确率的表格
    rankGraph(labels, summary_dir, picture_dir) # 生成分类器准确率的排名图形

def accExcel(stat, algorithms, output_dir, rank=10): # 生成预测准确率的表格
    if (os.path.exists(output_dir) == False):
        os.mkdir(output_dir)
    df = pd.DataFrame(stat, index=algorithms).T # 由于stat中存储的是 股票名-八种分类器准确率 的键值对， 故为了生成 分类器名 - 股票预测准确率 的键值对，需要对字典进行转置。
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Stock'}, inplace=True) # 将index列重命名为Stock
    dataframe = pd.DataFrame({'Rank': range(1, rank+1)}) # 新建一个DataFrame
    for algorithm in algorithms: # 遍历每一种分类器
        df_rank = df[['Stock', algorithm]].nlargest(rank, algorithm) # 选取分类器排名前rank的股票
        df_rank.reset_index(inplace=True)
        df_rank.rename(columns={'Stock': algorithm + '-Top ' + str(rank), algorithm: algorithm + ' Accuracy'}, inplace=True)
        dataframe = pd.concat([dataframe, df_rank], axis=1)
    dataframe.drop(columns=['index'], axis=1, inplace=True)
    dataframe.to_excel(output_dir + os.sep * 2 + 'TOP' + str(rank) + '.xlsx', index=False) # 生成分类器排名表格文件
    df.to_excel(output_dir + os.sep * 2 + 'Summary_prediction.xlsx',index=False)  # 生成预测汇总文件
    
def accGraph(stat, algorithms, markers,picture_dir):  #生成八种分类器在不同股票下的准确率折线图
    if (os.path.exists(picture_dir) == False):
        os.makedirs(picture_dir)
    x = []
    y = []
    for (key, value) in stat.items():
        x.append(key)
        y.append(value)
    y = np.array(y).T.tolist()
    plt.figure(figsize=(12, 8))
    plt.title('CVS - Train Set - Accuracy Comparison') # 生成训练集交叉验证准确率的图形
    for (li, type, marker) in zip(y, algorithms, markers):
        plt.plot(x, li, label=type, marker=marker)
    plt.hlines(y=0.5,xmin=stocks[0], xmax=stocks[len(stocks) - 1], colors='grey', linestyles='dashed') # 生成预测准确率50%的参考线
    plt.legend(algorithms) # 显示图例
    plt.savefig(picture_dir + os.sep * 2 + 'CVS-Acc-Summary.png', dpi=300)
    plt.close()

def rankGraph(algorithms, summary_dir, picture_dir): # 生成测试集预测平均准确率的排名图形
    rank_dir = picture_dir + os.sep * 2 + 'rankGraph'
    if (os.path.exists(rank_dir) == False):
        os.makedirs(rank_dir)
    file = summary_dir + os.sep * 2 + 'Summary_prediction.xlsx'
    df = pd.read_excel(file)
    dic = {}
    for algorithm in algorithms: # 对每一种分类器，绘制其测试集预测准确率的图形。
        plt.figure(figsize=(12, 8))
        average = df[algorithm].mean()
        dic[algorithm] = average
        plt.title(algorithm + ' - Test Set - Average Accuracy is ' + str(round(average, 3)))
        df_rank = df[['Stock', algorithm]].sort_values(algorithm, ascending=False)
        df_rank.reset_index(inplace=True)
        x = df_rank['Stock']
        height=df_rank[algorithm]
        plt.bar(x=x, height=height, label='Accuracy')
        for a, b in zip(x, height):
            plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=10)
        plt.hlines(y=average, xmin=df_rank['Stock'][0], xmax=df_rank['Stock'][df_rank['Stock'].size - 1],
        linestyles='dashed', colors='red', label='Average Accuracy')
        plt.xlabel('Stock')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(rank_dir + os.sep * 2 + algorithm + '_Rank.png', dpi=300)
        plt.close()
    
    plt.figure(figsize=(12, 8)) # 对于每一种分类器，统计在测试集上的平均预测准确率，绘制统计图形。
    x = sorted(dic, key=dic.__getitem__, reverse=True)
    height = [dic[i] for i in x]
    plt.title('Test Set - Mean Accuracy Rank')
    plt.bar(x=x, height=height, label='Mean Accuracy')
    for a, b in zip(x, height):
        plt.text(a, b + 0.01, '%.3f' % b, ha='center', va='bottom', fontsize=10)
    plt.legend()
    plt.savefig(picture_dir + os.sep * 2 + 'Mean-Acc-Rank.png', dpi=300)
    plt.close()
    

    
def distribution(stocks, input_dir, output_dir): #生成股票数据情感数据的分布图
    if (os.path.exists(output_dir) != True):
        os.mkdir(output_dir)
    for stock in stocks:
        df = pd.read_excel(input_dir + os.sep * 2 + stock + '_concat.xlsx')
        x = df['Compound_multiplied_scaled']
        y = df['Buy/Sell']
        plt.figure()
        plt.title(stock)
        plt.hist(x, density='true', bins=x.size)
        plt.xlabel('Compound_multiplied_scaled')
        plt.savefig(output_dir + os.sep * 2 + stock + '_histogram.png', dpi=300)
        plt.close()
    
def scatter(x, y): #生成股票情感归一化系数散点图
    index1 = np.argwhere(y == -1)
    index2 = np.argwhere(y == 1)
    plt.figure()
    plt.scatter(x[index2], y[index2], c='red', marker='^', label='Sell')
    plt.scatter(x[index1], y[index1], c='green', marker='v', label='Buy')
    plt.vlines(0, -1, 1, colors='grey', linestyles='dashed')
    plt.legend()
    plt.show()

def bestKNN(stock, x, y, start, end, graph, picture_dir): # 对KNN算法进行参数调优，参数区间范围是range(start, end)。
    output_dir = picture_dir + os.sep * 2 + 'bestKNN'
    if(os.path.exists(output_dir) == False):
        os.makedirs(output_dir)
    training_accuracy=[]
    test_accuracy = []
    k_score = []
    neighbors_range = range(start, end)

    for i in neighbors_range: # 对每一种参数下的预测准确率进行交叉验证，返回预测准确率最高的参数作为最优参数。
        clf = KNN(i)
        score = CVS(clf, x, y, cv=10)
        training_accuracy.append(score.mean())
        test_accuracy.append(score.mean())
        k_score.append(score.mean())
    
    if (graph == True):
        plt.figure(figsize=(12, 8))
        plt.title(stock)
        plt.plot(neighbors_range, k_score, label='CVS Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('n_neighbors')
        ax=plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1))
        plt.legend()
        plt.savefig(output_dir + os.sep * 2 + stock + '_bestKNN.png', dpi=300)
        plt.close()
    return neighbors_range[k_score.index(max(k_score))]

def getStockList(tweet_dir): #遍历Tweet目录下的所有推特文件，并返回文件名列表（即所有股票名）。
    stocks = []
    if (os.path.exists(tweet_dir) != True):
        return stocks
    else:
        for file in os.listdir(tweet_dir):
            filename = os.path.splitext(file)
            if filename[1] == '.xlsx':
                stocks.append(filename[0])
    return tuple(stocks)

def trade(stocks, algorithms, prediction_dir, trade_dir, tradeGraph_dir): # 交易函数，生成资产变化的图片与表格。
    if (os.path.exists(trade_dir) == False):
        os.makedirs(trade_dir)
    if (os.path.exists(tradeGraph_dir) == False):
        os.makedirs(tradeGraph_dir)
    for stock in stocks: # 对于每一支股票，使用(Close - Open) * 预测的Buy/Sell计算盈亏，最后累计每一个交易日盈亏作为资产变化。 
        df = pd.read_excel(prediction_dir + os.sep * 2 + stock + '_prediction.xlsx')
        first_Close = df.loc[0, 'Adj Close']
        for algorithm in algorithms:
            df[algorithm + '_Profit_or_Loss'] = (df['Close'] - df['Open']) * df[algorithm + '_prediction']
            df[algorithm + '_result'] = 0
            df.loc[0, algorithm + '_result'] = first_Close
            for i in range(1, len(df)):
                df.loc[i, algorithm + '_result'] = df.loc[i - 1, algorithm + '_result'] + df.loc[i, algorithm + '_Profit_or_Loss']
        df = df.drop(columns=['Unnamed: 0'])
        tradeGraph(stock, algorithms, df, 15, tradeGraph_dir) # 生成资产变化的曲线图
        df.to_excel(trade_dir + os.sep * 2 + stock + '_trade.xlsx', index=False) # 生成资产变化的表格

def tradeGraph(stock, algorithms, DataFrame, days, tradeGraph_dir):
    plt.style.use('fivethirtyeight') # 定义plt背景颜色风格
    plt.rcParams['figure.figsize'] = 16, 12 # 定义画布大小
    plt.figure() # 生成新画布
    plt.suptitle(stock, fontsize=24 , y=0.95) # 将股票名作为Title

    ax = []
    df = DataFrame
    ax.append(df['Adj Close']) # 将每一个交易日的Adj Close作为买入并持有策略的资产变化，存储到ax[0]中。
    colors = ('r', 'coral', 'green', 'orange', 'm', 'royalblue', 'saddlebrown', 'blueviolet', 'teal') # 定义曲线颜色
    plt.subplots_adjust(hspace=0.5, wspace=0.15) # 调整子图横竖间距，避免子图之间过于拥挤。
    for (i, algorithm) in enumerate(algorithms):
        ax.append(df[algorithm + '_result']) # 将不同算法每天的盈亏作为资产变化
        plt.subplot(4, 2, i+1)
        plt.plot(ax[0], colors[0], linewidth=2)
        plt.plot(ax[i+1], colors[i+1], linestyle='--', linewidth=3)
        plt.xlabel('Days', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        a = 'Buy-and-Hold'
        b = algorithm
        plt.title(a + ' vs. ' + b + ' strategy', fontsize=16)
        plt.legend((a, b), fontsize=10, loc='upper left', bbox_to_anchor=(0.05, 1),frameon=True)
    plt.savefig(tradeGraph_dir + os.sep * 2 + stock + '_tradeGraph.png') # 保存图片
    plt.close()
    

            
if __name__ == "__main__":
    tweet_dir = os.getcwd() + os.sep * 2 + 'Tweet'
    stock_dir = os.getcwd() + os.sep * 2 + 'Stocks'
    summary_dir = os.getcwd() + os.sep * 2 + 'Sheet'
    prediction_dir = summary_dir + os.sep * 2 + 'Prediction'
    picture_dir = os.getcwd() + os.sep * 2 + 'pics'
    distribution_dir = picture_dir + os.sep * 2 + 'Histogram'
    trade_dir = summary_dir + os.sep * 2 + 'Trade'
    tradeGraph_dir = picture_dir + os.sep * 2 + 'tradeGraph'
    algorithms = ('KNN', 'LogReg', 'NaiveBayes', 'SVM', 'DecisionTree', 'RandomForest', 'Perceptron', 'GBDT')
    prediction_days = 15
    stocks = getStockList(tweet_dir) # 根据推特文件获取股票名列表
    # stocks = ['AAL']
    scoreTweet(stocks, tweet_dir) # 对推特文件进行情感打分
    distribution(stocks, stock_dir, distribution_dir) # 生成情感得分分布图形
    getStocks(stocks, tweet_dir, stock_dir) # 获取股票文件，同时与情感得分文件进行合并。
    prediction(stocks, stock_dir, summary_dir, prediction_dir, picture_dir, prediction_days) # 使用八种分类器预测Buy/Sell值，输出预测表格文件。
    trade(stocks, algorithms, prediction_dir,trade_dir, tradeGraph_dir) # 应用交易策略，生成资产变化图形。
    
    


    


    
        
