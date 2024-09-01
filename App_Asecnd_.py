#importing libs
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression

#setting page configs
logo = Image.open("Image/Play Store Logo.png")
st.set_page_config(page_icon=logo,layout='wide',page_title='App Ascend')


#reading datafile
data = pd.read_csv("Data/Cleaned Apps Data.csv")
data.rename(columns={'Android_Supported':'Android Versions Supported','File_Size':'Application Size','App_Versions Categorical':'Application Verions','Downloads +':'Downloads'},inplace=True)
data['Update Recency'] = data['Recency'].apply(lambda x : abs(int(str(x).split(' ')[0])))
#page intoductio section
st.markdown("""
    <div style='text-align: justify;color:black;font-size:17px;'></b>App Ascend</b></div>""",unsafe_allow_html=True)
st.write('')
st.write("\n")
st.markdown("""
    <div style='text-align: justify;color:black;font-size:13px;'>The mobile application market has seen exponential growth over the past decade, with the Google Play Store emerging as a dominant platform for app distribution. Our analysis focuses on a dataset comprising roughly <b>10,000 mobile applications</b>, each updated <b>between 2010 and 2018</b>. This period captures a critical phase in the evolution of mobile apps, where the industry witnessed significant shifts in user preferences, technological advancements, and market dynamics. As the app ecosystem became increasingly competitive, understanding what drives an application\'s success became essential for developers and businesses alike. This study aims to uncover the key factors that contribute to the success of free mobile applications and to outline a strategic roadmap for launching a successful app.</div>""",unsafe_allow_html=True)
st.write("\n") 
st.markdown("""
<div style='text-align: justify;color:black;font-size:13px;'>The analysis addresses two core questions or problem statements:
<ul>
  <li style='font-size: 13px;'>What are the critical success factors for free apps on the Google Play Store?</li>
  <li style='font-size: 13px;'>How can these insights be leveraged to ensure a successful app launch?</li>
</ul></div>""",unsafe_allow_html=True)



########################### Section for Univariate Analysis ########################### 
###Categorial
univariate = st.container(border=True)
univariate.markdown("""
                    <div style='text-align: center;color:black;font-size:16px;'></b>Univariate Analysis</b></div>""",unsafe_allow_html=True)
univariate.write('\n')
#creating columns for chart and description
univariate_des, univariate_charts = univariate.columns([.35,.65])
univariate_des_ = univariate_des.container(border=True)
univariate_charts_ = univariate_charts.container(border=True)
#description
univariate_des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Before we dive deeper into determination of factors important to indicate success, we take a loot at the distribution of our mobile applications across various variables. This helps use in undersatdning the popular genres, underrated/unexplored genres, how are applications distributed across thier versions along with android versions supported.</div>""",unsafe_allow_html=True)
univariate_des_.write('\n')
univariate_des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Feel free to explore and play around and derive insights of your own, Some of our observations are given below:
<ul>
  <li style='font-size: 12px;'>Majority of the Applications are Free <b>97%</b></li>
  <li style='font-size: 12px;'><b>77%</b> of all the Applcations are Content Rated <b>Everyone</b></li>
</ul></div>""",unsafe_allow_html=True)
univariate_des_.write('\n')
variable_to_explore = univariate_des_.selectbox('Select a Variable to Explore!',['Application Size','Category','Content Rating',
                                                                                 'Genres','Android Versions Supported','Application Verions'])
#pie chart to show the distribution
pie_univariate = px.pie(values=data[variable_to_explore].value_counts().values,names=data[variable_to_explore].value_counts().index.tolist(),color=data[variable_to_explore].value_counts().index.to_list(),title=f'Distribution of Applications across {variable_to_explore}',height=450)
univariate_charts_.plotly_chart(pie_univariate,use_container_width=True)

###Continuos
un_con_chart ,un_con_des= univariate.columns([.80,.20])
un_con_des_ = un_con_des.container(border=True)
un_con_chart_ = un_con_chart.container(border=True)

#description
un_con_des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>After looking the the distribution of categorical variables let us also study the distribution and spread of continuos variables.</div>""",unsafe_allow_html=True)
un_con_des_.write('\n')
un_con_des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Feel free to explore and play around and derive insights of your own, Some of our observations are given below:
<ul>
  <li style='font-size: 12px;'>Significant skewness in almost all variables</li>
  <li style='font-size: 12px;'>Downloads, Update Recency, Reviews are all left skewed but Ratings is right skewed</li>
</ul></div>""",unsafe_allow_html=True)
un_con_des_.write('\n')
con_variable_to_explore = un_con_des_.selectbox('Select a Variable to Explore!',['Ratings','Downloads','Reviews','Update Recency'])

plots_den,plot_vio,plot_box = un_con_chart_.columns(3)
density_plot =  px.histogram(data, y=con_variable_to_explore, title=f"{con_variable_to_explore} Density Plot",histnorm='probability density')
violin = px.violin(data,y=con_variable_to_explore,title=f'ViolinPlot_{con_variable_to_explore}') 
box = px.box(data,y=con_variable_to_explore,title=f'Boxplot_{con_variable_to_explore}')
plots_den.plotly_chart(density_plot,use_container_width=True)
plot_vio.plotly_chart(violin,use_container_width=True)
plot_box.plotly_chart(box,use_container_width=True)

########################### Section for Bivariate Analysis ########################### 
bivariate = st.container(border=True)
bivariate.markdown("""
                    <div style='text-align: center;color:black;font-size:16px;'></b>Bivariate Analysis</b></div>""",unsafe_allow_html=True)
bivariate.write('\n')
#creating columns for chart and description
bivariate_des, bivariate_charts = bivariate.columns([.35,.65])
bivariate_des_ = bivariate_des.container(border=True)
bivariate_charts_ = bivariate_charts.container(border=True)

#description
bivariate_des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Now we dive deeper into how variables are interacting with each other 2 at a time. This helps in undersatdning how variables/entities stack up against each other given a certain KPI of your choice. You are once again free to play around with the charts below and compare and see for yourself how data unfolds.</div>""",unsafe_allow_html=True)
bivariate_des_.write('\n')
bivariate_des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Feel free to explore and play around and derive insights of your own, Some of our observations are given below:
<ul>
  <li style='font-size: 12px;'>Majority of the Applications are Free <b>97%</b></li>
  <li style='font-size: 12px;'><b>77%</b> of all the Applcations are Content Rated <b>Everyone</b></li>
</ul></div>""",unsafe_allow_html=True)
bivariate_des_.write('\n')

var_interest = bivariate_des_.selectbox('Select a Variable to Explore!',['Application Size','Category','Content Rating',
                                                                                 'Genres','Android Versions Supported','Application Verions'],key=33)
kpi_interest = bivariate_des_.selectbox('Select a Metric to Study!',['Ratings','Downloads','Reviews','Update Recency','Impact_Factor'],key=66)
method_interest = bivariate_des_.selectbox('Select a Metric to Study!',['Sum','Average','Frequency'],key=69)

#creating a definition to aggregate data
def aggregator(data,method_interest,kpi_interest,var_interest):
    if method_interest == 'Sum':
        agg = data.groupby([var_interest])[kpi_interest].sum()
    elif method_interest == 'Average':
        agg = data.groupby([var_interest])[kpi_interest].mean().round(1)
    elif method_interest == 'Frequency':
        agg = data.groupby([var_interest])['Application'].count()
    return agg

#creating a dataframe for aggregation of kpis over variables
agg = aggregator(data,method_interest,kpi_interest,var_interest)
agg = pd.DataFrame(agg.reset_index())
#plotting the bar chart for comparison
bi_chart = px.bar(pd.DataFrame(agg),color=var_interest,x=var_interest,
                        y=kpi_interest,title=f'{method_interest} {kpi_interest} over {var_interest}')
bivariate_charts_.plotly_chart(bi_chart)

#Sunburst chart
#creating columns for chart and description
bivariate_charts_sun,bivariate_des_sun  = bivariate.columns([.65,.35])
bivariate_des_sun_ = bivariate_des_sun.container(border=True)
bivariate_charts_sun_ = bivariate_charts_sun.container(border=True)

#description
bivariate_des_sun_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>A Sunburst chart is a type of data visualization that displays hierarchical data using a series of concentric rings. Each ring represents a level in the hierarchy, with the innermost circle being the root level and outer rings representing deeper levels. The size of each segment in the chart typically reflects a quantitative value, and the colors can help differentiate between categories or highlight specific data points.</div>""",unsafe_allow_html=True)
bivariate_des_sun_.write('\n')
bivariate_des_sun_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Feel free to explore and play around and derive insights of your own, Some of our observations are given below:
<ul>
  <li style='font-size: 12px;'>Majority of the Applications are Free <b>97%</b></li>
  <li style='font-size: 12px;'><b>77%</b> of all the Applcations are Content Rated <b>Everyone</b></li>
</ul></div>""",unsafe_allow_html=True)
bivariate_des_sun_.write('\n')
#variable selection for sunburst chart
var_1_list  = ['Application Size','Category','Content Rating','Genres','Android Versions Supported','Application Verions']
var_1 = bivariate_des_sun_.selectbox('Select a First Level Variable to Explore!',var_1_list,key=77)
var_2_list = [x for x in var_1_list if x != var_1]
var_2 = bivariate_des_sun_.selectbox('Select a Second Level Variable to Explore!',var_2_list,key=87)
kpi_dist = bivariate_des_sun_.selectbox('Select a Metric to Study!',['Downloads','Reviews'],key=99)

sun = px.sunburst(data, path=[var_1,var_2], 
                  values=kpi_dist,color=kpi_dist,template='plotly',hover_data=[var_1])
bivariate_charts_sun_.plotly_chart(sun)


## Correlation matrix
corr_data  = data[['Ratings','Reviews','Downloads','Purchase_Price','Update Recency','Impact_Factor','App_Versions Numeric','Android_Supported_Numeric']]
corr_data.rename(columns={'App_Versions Numeric':'App Versions','Android_Supported_Numeric':'Android Verions'},inplace=True)
size = pd.get_dummies(data['Application Size']).rename(columns={'Varies with device':'Dynamic'})
cat =  pd.get_dummies(data['Category'])
cone_rat = pd.get_dummies(data['Content Rating']).rename(columns={'Adults only 18+':'Adults 18+'})
gen = pd.get_dummies(data['Genres']).rename(columns={'Gaming and Interactive Content':'Gaming and IC','Entertainment and Leisure':'Ent. and Leisure','Learning and Education':'L & D','Lifestyle and Personal Care':'Lifestyle and PC'})

corr_data = pd.concat([corr_data,size],axis=1)
corr_data = pd.concat([corr_data,cat],axis=1)
corr_data = pd.concat([corr_data,cone_rat],axis=1)
corr_data = pd.concat([corr_data,gen],axis=1)


#plotting the correlation heatmap
corr_chart = bivariate.container(border=True)  
heatmap = px.imshow(corr_data.corr(method='spearman').round(2),labels=dict(x="Features", y="Features", color="Correlation"),aspect='auto',color_continuous_scale ='jet',text_auto=True,title='Correlation Heatmap for all the Variables',height=800)
heatmap.update_traces(textfont_size=11.5)
corr_chart.plotly_chart(heatmap,use_container_width=True)


########################### Mutli Variate ###########################
multivariate = st.container(border=True)
multivariate.markdown("""
                    <div style='text-align: center;color:black;font-size:16px;'></b>Multivariate Analysis</b></div>""",unsafe_allow_html=True)
multivariate.write('\n')

#creating columns for chart and description
multivariate_des, multivariate_charts = multivariate.columns([.35,.65])
multivariate_des_ = multivariate_des.container(border=True)
multivariate_charts = multivariate_charts.container(border=True)


#description
multivariate_des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>A Sunburst chart is a type of data visualization that displays hierarchical data using a series of concentric rings. Each ring represents a level in the hierarchy, with the innermost circle being the root level and outer rings representing deeper levels. The size of each segment in the chart typically reflects a quantitative value, and the colors can help differentiate between categories or highlight specific data points.</div>""",unsafe_allow_html=True)
multivariate_des_.write('\n')
multivariate_des_.markdown("""
                    <div style='text-align: justify;color:black;font-size:12px;'>Feel free to explore and play around and derive insights of your own, Some of our observations are given below:
<ul>
  <li style='font-size: 12px;'>Majority of the Applications are Free <b>97%</b></li>
  <li style='font-size: 12px;'><b>77%</b> of all the Applcations are Content Rated <b>Everyone</b></li>
</ul></div>""",unsafe_allow_html=True)
multivariate_des_.write('\n')

#select boxes for the variables that would be a part of the chart
var_1_list_m  = ['Application Size','Category','Content Rating','Genres','Android Versions Supported','Application Verions']
var_1_m = multivariate_des_.selectbox('Select a First Level Variable to Explore!',var_1_list_m,key=707)
var_2_list_m = [x for x in var_1_list if x != var_1_m]
var_2_m = multivariate_des_.selectbox('Select a Second Level Variable to Explore!',var_2_list_m,key=807)
kpi_dist_m = multivariate_des_.selectbox('Select a Metric to Study!',['Downloads','Reviews','Ratings','Impact_Factor'],key=909)
basis = multivariate_des_.selectbox('Select a Metric to Study!',['Sum','Average','Frequency'],key=1089)

multivariate_des_.write('\n')
#creating the datset that would be displayed
def level(basis):
    if basis=='Frequency':
        dem_subs = data.groupby([var_1_m,var_2_m])[kpi_dist_m].count()        
    elif basis =='Sum':
        dem_subs = data.groupby([var_1_m,var_2_m])[kpi_dist_m].sum()
    elif basis =='Sum':
        dem_subs = data.groupby([var_1_m,var_2_m])[kpi_dist_m].mean().round(1)
    return dem_subs

dem_subs=pd.DataFrame(level(basis))
dem_subs.reset_index(inplace=True)
#dem_subs.rename(columns={'Ref ID':'Number of Users'},inplace=True)


dem_subs_chart = px.bar(dem_subs,color=var_2_m,x=var_1_m,barmode='group',
                        y=kpi_dist_m)

multivariate_charts.plotly_chart(dem_subs_chart)



##### Model Building 
######
model_retention_rate = st.container(border=True)
model_retention_rate.markdown('<div style="text-align: center; font-size: 24px">Predictive Model on Applicaion Success</div>',unsafe_allow_html=True)
model_retention_rate.divider()
model_retention_rate_class,model_retention_rate_chart = model_retention_rate.columns([.35,.65]) 
model_retention_rate_class_ = model_retention_rate_class.container(border=True)
model_retention_rate_chart_ = model_retention_rate_chart.container(border=True)


#dataprep


model_data = corr_data.drop(['Ratings','Reviews','Downloads'],axis=1)
model_data['Success']  = model_data['Impact_Factor'].apply(lambda x : 1 if x >=.8 else 0)
model_data.drop(columns='Impact_Factor',inplace=True)
dependent = model_data['Success']
selected_vars = model_retention_rate_chart_.multiselect('Select Variables for the Model:',model_data.drop('Success',axis=1).columns.to_list(),default=model_data.drop('Success',axis=1).columns.to_list())

#setting smote and scaler
smt = SMOTE()
scaler = StandardScaler()
X_Scaled = scaler.fit_transform(model_data.drop(['Success'],axis=1)[selected_vars])
X, y = smt.fit_resample(X_Scaled,dependent)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
rfc = RandomForestClassifier()
model_rfc = rfc.fit(X_train,y_train)
y_pred_rfc = model_rfc.predict(X_test)



#visualisation of accracy
cm = confusion_matrix(y_test, y_pred_rfc)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=['Not Successful', 'Successfull'], y=['Not Successful', 'Successfull'], color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Predicted Customer Churn')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
model_retention_rate_class_.plotly_chart(con_mat,use_container_width=True)
model_retention_rate_class_.divider()
model_retention_rate_class_.text(classification_report(y_test,y_pred_rfc))


# Get feature importances
importances = model_rfc.feature_importances_

# Get feature names
feature_names = model_data[selected_vars].columns.to_list()
# Create a horizontal bar chart for feature importance
fig = go.Figure(go.Bar(
    x=feature_names,
    y=importances,
))

# Customize layout
fig.update_layout(
    title='Feature Importance in Random Forest Classifier',
    xaxis_title='Feature Names',
    yaxis_title='Importance',
)
model_retention_rate_chart_.plotly_chart(fig,use_container_width=True)






