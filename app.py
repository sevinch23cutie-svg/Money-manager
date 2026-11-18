import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title='AI Money Manager',page_icon='💸',layout='wide')
st.title('💸 AI Money Manager')
st.caption('Smart expense categorization and unusual spending detection · Educational demo')

examples = {
'Food':['starbucks coffee','grocery supermarket','restaurant lunch','pizza dinner','cafe latte','bread bakery','milk and eggs','chicken sandwich'],
'Transport':['bus fare','subway ticket','taxi ride','train ticket','transport card','metro fare','airport bus','fuel station'],
'Shopping':['new jacket','zara clothes','shoes purchase','online shopping','new backpack','cosmetics store','jeans purchase','electronics shop'],
'Housing':['monthly rent','apartment utilities','electricity bill','gas bill','water bill','housing deposit','building maintenance','home internet'],
'Entertainment':['netflix subscription','movie tickets','concert entry','spotify subscription','cinema popcorn','video game','amusement park','music streaming'],
'Education':['university textbook','tuition payment','online course','school supplies','language class','exam fee','study materials','notebook for class'],
'Health':['pharmacy medicine','hospital visit','doctor consultation','dental checkup','vitamins purchase','clinic bill','medical test','prescription medicine']}
texts=[x for items in examples.values() for x in items]
labels=[category for category,items in examples.items() for _ in items]
@st.cache_resource
def train():
    model=make_pipeline(TfidfVectorizer(ngram_range=(1,2),analyzer='char_wb',min_df=1),LogisticRegression(max_iter=1500))
    model.fit(texts,labels)
    return model
model=train()

if 'expenses' not in st.session_state:
    st.session_state.expenses=pd.DataFrame([
        ('2026-11-01','Monthly rent',450000,'Housing'),('2026-11-02','Grocery supermarket',48000,'Food'),
        ('2026-11-03','Bus fare',1600,'Transport'),('2026-11-04','Starbucks coffee',5500,'Food'),
        ('2026-11-05','Netflix subscription',17000,'Entertainment'),('2026-11-06','University textbook',32000,'Education'),
        ('2026-11-07','Grocery supermarket',62000,'Food'),('2026-11-08','Taxi ride',13500,'Transport'),
        ('2026-11-09','Restaurant lunch',12000,'Food'),('2026-11-10','Pharmacy medicine',18000,'Health'),
        ('2026-11-11','New jacket',89000,'Shopping'),('2026-11-12','Cafe latte',4800,'Food'),
    ],columns=['Date','Description','Amount','Category'])

with st.sidebar:
    st.header('Settings')
    currency=st.selectbox('Display currency (no conversion)',['KRW (₩)','USD ($)'])
    symbol='₩' if currency.startswith('KRW') else '$'
    st.info('Amounts stay in the selected currency. Switching the label does not convert values.')
    st.markdown('**AI models:** TF-IDF + Logistic Regression; Isolation Forest')
    st.caption('Sample data is fictional. No bank connection.')

st.subheader('Add an expense')
with st.form('add'):
    a,b,c=st.columns([2,1,1])
    desc=a.text_input('Description',placeholder='e.g. Starbucks coffee')
    amount=b.number_input('Amount',min_value=0.0,step=100.0)
    date=c.date_input('Date')
    submitted=st.form_submit_button('✨ Classify & add expense',use_container_width=True)
    if submitted:
        if not desc.strip() or amount<=0:
            st.error('Enter a description and an amount greater than zero.')
        else:
            pred=model.predict([desc])[0]
            confidence=float(model.predict_proba([desc]).max())
            st.session_state.expenses=pd.concat([st.session_state.expenses,pd.DataFrame([{'Date':str(date),'Description':desc.strip(),'Amount':amount,'Category':pred}])],ignore_index=True)
            st.success(f'AI predicted: {pred} (model confidence {confidence:.0%}). You can correct categories below.')

st.subheader('Or upload a CSV')
upload=st.file_uploader('CSV columns: Date, Description, Amount (Category optional)',type='csv')
if upload:
    try:
        incoming=pd.read_csv(upload)
        required={'Date','Description','Amount'}
        if not required.issubset(incoming.columns):
            st.error('CSV must contain Date, Description, and Amount columns.')
        else:
            incoming=incoming.copy()
            incoming['Amount']=pd.to_numeric(incoming['Amount'],errors='coerce')
            incoming=incoming.dropna(subset=['Description','Amount'])
            incoming=incoming[incoming['Amount']>0]
            if 'Category' not in incoming.columns:
                incoming['Category']=model.predict(incoming['Description'].astype(str)) if len(incoming) else []
            if st.button(f'Import {len(incoming)} valid rows'):
                st.session_state.expenses=pd.concat([st.session_state.expenses,incoming[['Date','Description','Amount','Category']]],ignore_index=True)
                st.rerun()
    except Exception as exc:
        st.error(f'Could not read CSV: {exc}')

df=st.session_state.expenses.copy()
if not df.empty:
    total=df['Amount'].sum()
    top=df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    m1,m2,m3=st.columns(3)
    m1.metric('Total expenses',f'{symbol}{total:,.0f}')
    m2.metric('Transactions',len(df))
    m3.metric('Largest category',top.index[0])
    left,right=st.columns(2)
    with left:
        st.plotly_chart(px.pie(df,names='Category',values='Amount',hole=.48,title='Spending by category'),use_container_width=True)
    with right:
        chart=df.groupby('Category',as_index=False)['Amount'].sum().sort_values('Amount',ascending=False)
        st.plotly_chart(px.bar(chart,x='Category',y='Amount',title='Category totals'),use_container_width=True)
    st.subheader('🔎 Unusual spending')
    st.caption('Isolation Forest flags statistical outliers in transaction amounts (log scale). Flags are not financial advice.')
    if len(df)>=8 and df['Amount'].nunique()>=3:
        import numpy as np
        detector=IsolationForest(contamination=.15,random_state=42)
        df['Unusual']=detector.fit_predict(np.log1p(df[['Amount']]))==-1
        flagged=df[df['Unusual']]
        if len(flagged):
            st.warning(f'{len(flagged)} transaction(s) flagged as unusual for review.')
            st.dataframe(flagged[['Date','Description','Amount','Category']],hide_index=True,use_container_width=True)
        else: st.success('No unusual transactions detected.')
    else: st.info('At least 8 transactions with varied amounts are needed for anomaly detection.')
    st.subheader('Transactions — edit categories or amounts')
    edited=st.data_editor(df[['Date','Description','Amount','Category']],num_rows='dynamic',hide_index=True,use_container_width=True,key='editor',column_config={'Category':st.column_config.SelectboxColumn('Category',options=list(examples.keys())+['Other'])})
    if st.button('Save table edits'):
        edited['Amount']=pd.to_numeric(edited['Amount'],errors='coerce')
        if edited[['Date','Description','Amount','Category']].isna().any().any() or (edited['Amount']<=0).any():
            st.error('All rows need valid fields and positive amounts.')
        else:
            st.session_state.expenses=edited.reset_index(drop=True)
            st.rerun()
    st.download_button('Download expenses CSV',df[['Date','Description','Amount','Category']].to_csv(index=False).encode(),file_name='expenses.csv',mime='text/csv')
    st.info(f'💡 Insight: {top.index[0]} accounts for {top.iloc[0]/total:.0%} of recorded expenses. Review whether this matches your priorities.')
else:
    st.info('Add an expense to start.')
st.divider()
st.caption('Limitations: The small English training set may misclassify unfamiliar merchants; anomalies are relative to recorded amounts, not proof of waste. Data is stored only in this browser session and may disappear on refresh or server restart.')
