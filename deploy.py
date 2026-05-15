import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

model=pickle.load(open('gmm.pkl','rb'))
scale=pickle.load(open('scaler.pkl','rb'))
pca=pickle.load(open('pca.pkl','rb'))
df=pd.read_excel('World_development_mesurement.xlsx')

df['Business Tax Rate']=df['Business Tax Rate'].str.replace('%','')
df['Business Tax Rate']=df['Business Tax Rate'].astype('float')
df['Business Tax Rate (%)']=df['Business Tax Rate']/100
df.drop(columns='Business Tax Rate',inplace=True)

df['CO2 Emissions (in Thousands)']=df['CO2 Emissions']/1000
df.drop(columns='CO2 Emissions',inplace=True)

df['Energy Usage (in Thousands)']=df['Energy Usage']/1000
df.drop(columns='Energy Usage',inplace=True)

df['GDP']=df['GDP'].str.replace('$','').str.replace(',','')
df['GDP']=df['GDP'].astype('float')
df['GDP (in Millions)']=df['GDP']/1000000
df.drop(columns='GDP',inplace=True)

df['Health Exp/Capita']=df['Health Exp/Capita'].str.replace('$','').str.replace(',','')
df['Health Exp/Capita']=df['Health Exp/Capita'].astype('float')

df.drop(columns='Number of Records',inplace=True)

df['Population Total (in Millions)']=df['Population Total']/1000000
df.drop(columns='Population Total',inplace=True)

df['Tourism Inbound']=df['Tourism Inbound'].str.replace('$','').str.replace(',','')
df['Tourism Inbound']=df['Tourism Inbound'].astype('float')
df['Tourism Inbound (in Millions)']=df['Tourism Inbound']/1000000
df.drop(columns='Tourism Inbound',inplace=True)

df['Tourism Outbound']=df['Tourism Outbound'].str.replace('$','').str.replace(',','')
df['Tourism Outbound']=df['Tourism Outbound'].astype('float')
df['Tourism Outbound (in Millions)']=df['Tourism Outbound']/1000000
df.drop(columns='Tourism Outbound',inplace=True)

df['Years']=df.groupby('Country')['Country'].rank(method='first')

def country_report(country):
  '''
  Just enter the name of the country and you will get a complete report of all various factors to compare. If data is not available the chart will mention it.
  '''
  x=df[df['Country']==country]
  fig=plt.figure(figsize=(12,20))
  plt.suptitle(f'Report of {country}\n ',size=20,color='red')

  plt.subplot(5,2,1)
  x['urban']=x['Population Total (in Millions)']*x['Population Urban']
  if x[['urban','Population Total (in Millions)']].isnull().sum().sum()>10:
    plt.text(7,0.5,'Data is not Available',size=15,ha='center',va='center')
  else:
    plt.plot(x['Years'],x['Population Total (in Millions)'],lw=5,color='blue')
    plt.fill_between(x['Years'],x['Population Total (in Millions)'],color='blue',alpha=0.2)
    plt.plot(x['Years'],x['urban'],label='Urban',lw=5,color='green')
    plt.fill_between(x['Years'],x['urban'],color='green',alpha=0.2)
  plt.title('Population Distribution',size=20,color='red')
  plt.ylabel('in Millions',color='blue')
  plt.xlabel('Years',color='blue')
  plt.xlim(0,14)
  plt.legend()

  plt.subplot(5,2,2)
  if x[['Population 0-14','Population 15-64','Population 65+']].isnull().sum().sum()>10:
    plt.text(7,0.5,'Data is not Available',size=15,ha='center',va='center')
  else:
    plt.plot(x['Years'],x['Population 0-14'],label='0-14',lw=5,color='yellow')
    plt.fill_between(x['Years'],x['Population 0-14'],color='yellow',alpha=0.2)
    plt.plot(x['Years'],x['Population 15-64'],label='15-64',lw=5,color='green')
    plt.fill_between(x['Years'],x['Population 15-64'],color='green',alpha=0.2)
    plt.plot(x['Years'],x['Population 65+'],label='65+',lw=5,color='grey')
    plt.fill_between(x['Years'],x['Population 65+'],color='grey',alpha=0.2)
  plt.title('Demographic',size=20,color='red')
  plt.ylabel('Percent',color='blue')
  plt.xlabel('Years',color='blue')
  plt.xlim(0,14)
  plt.legend()

  plt.subplot(5,2,3)
  if x[['Birth Rate','Infant Mortality Rate']].isnull().sum().sum()>10:
    plt.text(7,0.5,'Data is not Available',size=15,ha='center',va='center')
  else:
    plt.plot(x['Years'],x['Birth Rate'],label='Birth Rate',lw=5,color='yellow')
    plt.fill_between(x['Years'],x['Birth Rate'],alpha=0.4,color='yellow')
    plt.plot(x['Years'],x['Infant Mortality Rate'],label='infant Mortality',lw=5,color='red')
  plt.title('Child Mortality',size=20,color='red')
  plt.xlim(0,14)
  plt.ylabel('Per 1000',color='blue')
  plt.xlabel('Years',color='blue')
  plt.legend()

  plt.subplot(5,2,4)
  if x[['Life Expectancy Female','Life Expectancy Male']].isnull().sum().sum()>10:
    plt.text(7,0.5,'Data is not Available',size=15,ha='center',va='center')
  else:
    plt.plot(x['Years'],x['Life Expectancy Female'],lw=5,label=f'Female',color='pink')
    plt.plot(x['Years'],x['Life Expectancy Male'],lw=5,label=f'Male',color='blue')
  plt.title('Life Expectancy',size=20,color='red')
  plt.ylabel('Life Expectency',color='blue')
  plt.xlim(0,14)
  plt.xlabel('Years',color='blue')
  plt.legend()

  plt.subplot(5,2,5)
  x['health']=x['GDP (in Millions)']*x['Health Exp % GDP']
  if x[['GDP (in Millions)','health']].isnull().sum().sum()>10:
    plt.text(7,0.5,'Data is not Available',size=15,ha='center',va='center')
  else:
    plt.plot(x['Years'],x['GDP (in Millions)'],lw=5,color='blue')
    plt.fill_between(x['Years'],x['GDP (in Millions)'],color='blue',alpha=0.2)
    plt.plot(x['Years'],x['health'],label='Health',lw=5,color='green')
    plt.fill_between(x['Years'],x['health'],color='green',alpha=0.2)
  plt.title('Health Expense by GDP',size=20,color='red')
  plt.xlim(0,14)
  plt.ylabel('in Millions',color='blue')
  plt.xlabel('Years',color='blue')
  plt.legend()

  plt.subplot(5,2,6)
  if x[['Tourism Inbound (in Millions)','Tourism Outbound (in Millions)']].isnull().sum().sum()>10:
    plt.text(7,0.5,'Data is not Available',size=15,ha='center',va='center')
  else:
    plt.plot(x['Years'],x['Tourism Inbound (in Millions)'],label='Inflow',lw=5,color='blue')
    plt.plot(x['Years'],x['Tourism Outbound (in Millions)'],label='Outflow',lw=5,color='red')
    plt.fill_between(x['Years'],x['Tourism Inbound (in Millions)'],x['Tourism Outbound (in Millions)'],alpha=0.4,color='grey')
  plt.title('Toursim',size=20,color='red')
  plt.ylabel('in Millions',color='blue')
  plt.xlim(0,14)
  plt.xlabel('Years',color='blue')
  plt.legend()

  plt.subplot(5,2,7)
  x['hours to start buisness']=x['Days to Start Business']*24
  if x[['hours to start buisness','Hours to do Tax']].isnull().sum().sum()>10:
    plt.text(7,0.5,'Data is not Available',size=15,ha='center',va='center')
  else:
    plt.plot(x['Years'],x['hours to start buisness'],lw=5,color='blue',label='To Start Buisness')
    plt.fill_between(x['Years'],x['hours to start buisness'],color='blue',alpha=0.2)
    plt.plot(x['Years'],x['Hours to do Tax'],label='To do Tax',lw=5,color='green')
    plt.fill_between(x['Years'],x['Hours to do Tax'],color='green',alpha=0.2)
  plt.title(f'Buisness Metrics',size=20,color='red')
  plt.text(8,x['hours to start buisness'].max()-(x['hours to start buisness'].max()*1/4),f'Ease of Doing Buisness Rank\n{x["Ease of Business"].iloc[-1]}',size=10,color='red')
  plt.ylabel('Hours',color='blue')
  plt.xlim(0,14)
  plt.xlabel('Years',color='blue')
  plt.legend()

  plt.subplot(5,2,8)
  if x[['Lending Interest','Business Tax Rate (%)']].isnull().sum().sum()>10:
    plt.text(7,0.5,'Data is not Available',size=15,ha='center',va='center')
  else:
    plt.plot(x['Years'],x['Lending Interest'],label='Interest Rates',lw=5)
    plt.fill_between(x['Years'],x['Lending Interest'],alpha=0.2)
    plt.plot(x['Years'],x['Business Tax Rate (%)'],label='Tax Rates',lw=5,color='green')
    plt.fill_between(x['Years'],x['Business Tax Rate (%)'],alpha=0.5,color='green')
  plt.title('Intrest and Taxes',size=20,color='red')
  plt.ylabel('Percent',color='blue')
  plt.xlim(0,14)
  plt.xlabel('Years',color='blue')
  plt.legend()

  plt.subplot(5,2,9)
  if x[['Mobile Phone Usage','Internet Usage']].isnull().sum().sum()>10:
    plt.text(7,0.5,'Data is not Available',size=15,ha='center',va='center')
  else:
    plt.plot(x['Years'],x['Mobile Phone Usage'],label='Mobile Usage',lw=5)
    plt.fill_between(x['Years'],x['Mobile Phone Usage'],alpha=0.2)
    plt.plot(x['Years'],x['Internet Usage'],label='Internet Usage',lw=5,color='green')
    plt.fill_between(x['Years'],x['Internet Usage'],alpha=0.5,color='green')
  plt.title('Digital FootPrint',size=20,color='red')
  plt.ylabel('Usage (in %)',color='blue')
  plt.xlabel('Years',color='blue')
  plt.xlim(0,14)
  plt.legend()

  plt.subplot(5,2,10)
  if x[['CO2 Emissions (in Thousands)','Energy Usage (in Thousands)']].isnull().sum().sum()>10:
    plt.text(7,0.5,'Data is not Available',size=15,ha='center',va='center')
  else:
    plt.plot(x['Years'],x['CO2 Emissions (in Thousands)'],label='Emissions',lw=5,color='grey')
    plt.fill_between(x['Years'],x['CO2 Emissions (in Thousands)'],color='grey',alpha=0.25)
    plt.plot(x['Years'],x['Energy Usage (in Thousands)'],label='Energy Consumption',lw=5,color='red')
  plt.title('Environmental FootPrint',size=20,color='red')
  plt.xlim(0,14)
  plt.ylabel('Emission / Usage (in Thousands)',color='blue')
  plt.xlabel('Years',color='blue')
  plt.legend()

  plt.tight_layout()
  return fig

st.set_page_config(layout='wide')
st.title('GLobal Development Project')
tab1,tab2=st.tabs(['Country Report','Cluster Prediction'])

df_com=df.drop(columns=['Years','Ease of Business'])
df_coms=df_com.groupby('Country').median().reset_index()
num=df_coms.drop(columns=('Country')).columns
df_com=df_coms[num].fillna(df_coms[num].median())

with tab1:
  c1,c2=st.columns([1,4])
  with c1:
    country=st.selectbox('Select Country',sorted(df['Country'].unique()))
    if st.button('Generate Report'):
      with c2:
        fig=country_report(country)
        st.pyplot(fig)
      with c1:
        st.subheader(f'{country} Stats')
        stats=df_com[df_coms['Country']==country].T.reset_index()
        stats.columns=['Indicator','Value']
        stats=stats.iloc[1:-1]
        # st.dataframe(stats,use_container_width=True,hide_index=True)
        st.table(stats)
with tab2:
  def out_cap(df,c):
    q1=df[c].quantile(0.25)
    q3=df[c].quantile(0.75)
    iqr=q3-q1
    ue=q3+1.5*iqr
    le=q1-1.5*iqr
    df[c]=df[c].apply(lambda x:ue if x>ue else le if x<le else x)
  def inputs():
    c1,c2,c3,c4,c5,c6=st.columns([1,1,1,1,1,1])
    with c1:
      br=st.number_input('Birth Rate (per 1000)',step=0.001,format="%.3f")
      buis=st.number_input('Number of days to start a Buiseness',step=1,format="%d")
      hgdp=st.number_input('Percent of GDP on Healthcare (%)',step=0.01,format="%.2f")
    with c2:
      hpc=st.number_input('Health Expense per Capita',step=1,format="%d")
      hrs=st.number_input('Hours to do Tax',step=1,format="%d")
      ifr=st.number_input('Infant Mortality Rate (per 1000)',step=0.001,format="%.3f")
      inu=st.number_input('Internet Usage (%)',step=0.01,format="%.2f")
    with c3:
      lr=st.number_input('Lending Interest (%)',step=0.01,format="%.2f")
      lef=st.number_input('Life Expectancy of Female',step=1,format="%d")
      lem=st.number_input('Life Expectancy of Male',step=1,format="%d")
      mu=st.number_input('Mobile Usage (%)',step=0.01,format="%.2f")
    with c4:
      p1=st.number_input('Population 0-14 (%)',step=0.01,format="%.2f")
      p2=st.number_input('Population 15-64 (%)',step=0.01,format="%.2f")
      p3=st.number_input('Population 65+ (%)',step=0.01,format="%.2f")
      pu=st.number_input('Urban Population (%)',step=0.01,format="%.2f")
    with c5:
      bt=st.number_input('Buisness Tax Rate (%)',step=0.01,format="%.2f")
      co2=st.number_input('Co2 Emissions (in Thousands)',step=1,format="%d")
      ene=st.number_input('Energy Usage (in Thousands)',step=1,format="%d")
      gdp=st.number_input('GDP (in Millions)',step=1,format="%d")
    with c6:
      pt=st.number_input('Total Population (in Millions)',step=1,format="%d")
      ti=st.number_input('Tourism Inflow (in Millions)',step=1,format="%d")
      to=st.number_input('Tourism Outflow (in Millions)',step=1,format="%d")
    dic={'Birth Rate':br,'Days to Start Business':buis,'Health Exp % GDP':hgdp,'Health Exp/Capita':hpc,'Hours to do Tax':hrs,'Infant Mortality Rate':ifr,'Internet Usage':inu,'Lending Interest':lr,'Life Expectancy Female':lef,'Life Expectancy Male':lem,'Mobile Phone Usage':mu,'Population 0-14':p1,'Population 15-64':p2,'Population 65+':p3,'Population Urban':pu,'Business Tax Rate (%)':bt,'CO2 Emissions (in Thousands)':co2,'Energy Usage (in Thousands)':ene,'GDP (in Millions)':gdp,'Population Total (in Millions)':pt,'Tourism Inbound (in Millions)':ti,'Tourism Outbound (in Millions)':to}
    feature=pd.DataFrame(dic,index=[0])
    cap_col=['CO2 Emissions (in Thousands)','Energy Usage (in Thousands)','GDP (in Millions)','Population Total (in Millions)','Tourism Inbound (in Millions)','Tourism Outbound (in Millions)']
    for x in feature[cap_col]:
      out_cap(feature,x)
    feature=pd.DataFrame(scale.transform(feature),columns=feature.columns)
    feature=pd.DataFrame(pca.transform(feature))
    return feature
  dff=inputs()
  from sklearn.metrics import silhouette_score
  st.divider()
  # df_com=df_com.drop(columns='Country')
  cap_col=['CO2 Emissions (in Thousands)','Energy Usage (in Thousands)','GDP (in Millions)','Population Total (in Millions)','Tourism Inbound (in Millions)','Tourism Outbound (in Millions)']
  for x in df_com[cap_col]:
    out_cap(df_com,x)
  df_s=pd.DataFrame(scale.fit_transform(df_com),columns=df_com.columns)
  df_com=df_s.copy()
  df_com=pd.DataFrame(pca.fit_transform(df_com))
  label=model.fit_predict(df_com)
  score=silhouette_score(df_com,label)
  df_com['cluster']=label
  df_o=pd.DataFrame(scale.inverse_transform(df_s),columns=df_s.columns)
  df_o['cluster']=label
  df_o['country']=df_coms['Country']

  c1,c2=st.columns([1,4])
  with c1:
      but=st.button("Prediction")
      pred=model.predict(dff)
      if but:
        st.write('The Indicator classify the Nation as')
        if pred[0]==2:
          st.write(f'Cluster {pred[0]} [Highly Developed]')
        elif pred[0]==5:
          st.write(f'Cluster {pred[0]} [Developed]')
        elif pred[0]==4:
          st.write(f'Cluster {pred[0]} [New Developing]')
        elif pred[0]==7:
          st.write(f'Cluster {pred[0]} [Emerging]')
        elif pred[0]==3:
          st.write(f'Cluster {pred[0]} [Transitioning]')
        elif pred[0]==0:
          st.write(f'Cluster {pred[0]} [Developing]')
        elif pred[0]==6:
          st.write(f'Cluster {pred[0]} [Under Developed]')
        elif pred[0]==1:
          st.write(f'Cluster {pred[0]} [Least Developed]')
        st.write('Similar Countries Include')
        c=df_o[df_o['cluster']==pred[0]]['country'].tolist()
        t=' / '.join(c)
        st.write(t)
        with c2:
          df_0=df_com[df_com['cluster']==0]
          df_1=df_com[df_com['cluster']==1]
          df_2=df_com[df_com['cluster']==2]
          df_3=df_com[df_com['cluster']==3]
          df_4=df_com[df_com['cluster']==4]
          df_5=df_com[df_com['cluster']==5]
          df_6=df_com[df_com['cluster']==6]
          df_7=df_com[df_com['cluster']==7]
          fig=plt.figure(figsize=(12,6))
          plt.scatter(df_2[0],df_2[1],label='Highly Developed',alpha=0.75,c='deepskyblue')
          plt.scatter(df_5[0],df_5[1],label='Developed',alpha=0.75,c='darkgrey')
          plt.scatter(df_4[0],df_4[1],label='New Developing',alpha=0.75,c='black')
          plt.scatter(df_7[0],df_7[1],label='Emerging',alpha=0.75,c='magenta')
          plt.scatter(df_3[0],df_3[1],label='Transitioning',alpha=0.75,c='gold')
          plt.scatter(df_0[0],df_0[1],label='Developing',alpha=0.75,c='tomato')
          plt.scatter(df_6[0],df_6[1],label='Under Developed',alpha=0.75,c='chocolate')
          plt.scatter(df_1[0],df_1[1],label='Least Developed',alpha=0.75,c='lawngreen')
          plt.scatter(dff.iloc[:,0],dff.iloc[:,1],marker='*',s=50,label='Prediction')
          plt.title(f'Silhoutte score = {score}')
          plt.legend()
          st.pyplot(fig)
