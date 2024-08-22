from portfolio_optimization.hierarchical_risk_parity.hierarchial_optimizer \
    import generateData, plotCorrMatrix, correlDist, \
        getQuasiDiag, getRecBipart

import scipy.cluster.hierarchy as sch

from get_data.read_data import get_bulk_ticker_data_pak

import json

# load config file to extract necessary information
with open('config/tickers.json') as f:
    configs = json.load(f)
    f.close()

kse100 = configs['kse100']

df = get_bulk_ticker_data_pak(kse100, '2024-01-01', '2024-04-01')
df = df[['Date', 'Ticker', 'Close']]

df_pvt = df.pivot(index='Date', columns='Ticker', values='Close')

df_pvt = df_pvt.ffill()
df_pvt = df_pvt.bfill()

daily_return = df_pvt.pct_change(1).dropna()
cov,corr=daily_return.cov(),daily_return.corr()
plotCorrMatrix('HRP3_corr0.png',corr,labels=corr.columns)
dist=correlDist(corr)
link=sch.linkage(dist,'single')
sortIx=getQuasiDiag(link)
sortIx=corr.index[sortIx].tolist() # recover labels
df0=corr.loc[sortIx,sortIx] # reorder
plotCorrMatrix('HRP3_corr1.png',df0,labels=df0.columns)
#4) Capital allocation
hrp=getRecBipart(cov,sortIx)
print(hrp)
hrp.to_csv('portfolio_optimization/hierarchical_risk_parity/allocations.csv')

def main():
    #1) Generate correlated data
    nObs,size0,size1,sigma1=10000,5,5,.25
    x,cols=generateData(nObs,size0,size1,sigma1)
    print([(j+1,size0+i) for i,j in enumerate(cols,1)])
    cov,corr=x.cov(),x.corr()
    #2) compute and plot correl matrix
    plotCorrMatrix('HRP3_corr0.png',corr,labels=corr.columns)
    #3) cluster
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist() # recover labels
    df0=corr.loc[sortIx,sortIx] # reorder
    plotCorrMatrix('HRP3_corr1.png',df0,labels=df0.columns)
    #4) Capital allocation
    hrp=getRecBipart(cov,sortIx)
    print(hrp)
    return


if __name__=='__main__':
    main()


from hierarchical_risk_parity.hierarchial_optimizer \
    import generateData, plotCorrMatrix, correlDist, \
        getQuasiDiag, getRecBipart

import scipy.cluster.hierarchy as sch

def main():
    #1) Generate correlated data
    nObs,size0,size1,sigma1=10000,5,5,.25
    x,cols=generateData(nObs,size0,size1,sigma1)
    print([(j+1,size0+i) for i,j in enumerate(cols,1)])
    cov,corr=x.cov(),x.corr()
    #2) compute and plot correl matrix
    plotCorrMatrix('HRP3_corr0.png',corr,labels=corr.columns)
    #3) cluster
    dist=correlDist(corr)
    link=sch.linkage(dist,'single')
    sortIx=getQuasiDiag(link)
    sortIx=corr.index[sortIx].tolist() # recover labels
    df0=corr.loc[sortIx,sortIx] # reorder
    plotCorrMatrix('HRP3_corr1.png',df0,labels=df0.columns)
    #4) Capital allocation
    hrp=getRecBipart(cov,sortIx)
    print(hrp)
    return


if __name__=='__main__':
    main()


