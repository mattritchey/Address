# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 07:26:42 2022

@author: mritchey
"""
# streamlit run "C:\Users\mritchey\.spyder-py3\Python Scripts\streamlit projects\quick address\quick_address.py"
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import folium
from joblib import Parallel, delayed


@st.cache
def convert_df(df):
    return df.to_csv(index=0).encode('utf-8')


def map_results(results):
    for index, row in results.iterrows():
        address, sq_ft = results.loc[index,
                                     'Address'], results.loc[index, 'Total Area']
        html = f"""<p style="arial"><p style="font-size:14px"> 
                {address}  
                <br> Square Footage: {sq_ft}"""

        iframe = folium.IFrame(html)
        popup = folium.Popup(iframe,
                             min_width=140,
                             max_width=140)

        folium.Marker(location=[results.loc[index, 'Lat'],
                                results.loc[index, 'Lon']],
                      fill_color='#43d9de',
                      popup=popup,
                      radius=8).add_to(m)
    return folium


@st.cache
def get_housing_data(address_input):
    address = address_input.replace(
        ' ', '+').replace(',', '').replace('#+', '').upper()
    try:
        census = pd.read_json(
            f"https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress?address={address}&benchmark=2020&vintage=2020&format=json")
        results = census.iloc[:1, 0][0]
        matchedAddress_first = results[0]['matchedAddress']
        matchedAddress_last = results[-1]['matchedAddress']
        lat, lon = results[0]['coordinates']['y'], results[0]['coordinates']['x']
        # lat2, lon2 = results[-1]['coordinates']['y'], results[-1]['coordinates']['x']
        censusb = pd.DataFrame({'Description': ['Address Input', 'Census Matched Address: First',
                                                'Census Matched Address: Last', 'Lat', 'Lon'],
                                'Values': [address_input, matchedAddress_first, matchedAddress_last, lat, lon]})

        #Property Records
        url = f'https://www.countyoffice.org/property-records-search/?q={address}'
        county_office_list = pd.read_html(url)

        if county_office_list[1].shape[1] == 2:
            df2 = pd.concat([county_office_list[0], county_office_list[1]])
        else:
            df2 = county_office_list[0]
        df2.columns = ['Description', 'Values']

        final = censusb.append(df2)

        #Transpose
        final2 = final.T
        final2.columns = final2.loc['Description']
        final2 = final2.loc[['Values']].set_index('Address Input')
        # final2['County Office Url']=url
    except:
        final2 = address_input
    return final2


@st.cache(allow_output_mutation=True)
def address_quick(df, n_jobs=24):
    if isinstance(df, pd.DataFrame):
        df = df.drop_duplicates()
        df['address_input'] = df.iloc[:, 0]+', '+df.iloc[:, 1] + \
            ', '+df.iloc[:, 2]+' '+df.iloc[:, 3].astype(str).str[:5]
        df['address'] = df['address_input'].replace(
            {' ': '+', ',': ''}, regex=True).str.upper()
        df['address'] = df['address'].replace({'#+': ''}, regex=True)
        # addresses=df['address'].values
        addresses_input = df['address_input'].values
    else:
        addresses_input = [df]
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(get_housing_data)(i) for i in addresses_input)
    results_df = [i for i in results if isinstance(i, pd.DataFrame)]
    results_errors = [i for i in results if not isinstance(i, pd.DataFrame)]
    errors = pd.DataFrame({'Error Addresses': results_errors})
    final_results = pd.concat(results_df)
    final_results = final_results[final_results.columns[2:]].copy()

    return final_results, errors


st.set_page_config(layout="wide")
col1, col2 = st.columns((2))

address = st.sidebar.text_input(
    "Address", "1500 MOHICAN DR, FORESTDALE, AL, 35214")
uploaded_file = st.sidebar.file_uploader("Choose a file")
uploaded_file = 'C:/Users/mritchey/addresses_sample.csv'
address_file = st.sidebar.radio('Choose',
                                ('Single Address', 'Addresses (Geocode: Will take a bit)'))


if address_file == 'Addresses (Geocode: Will take a bit)':
    try:
        df = pd.read_csv(uploaded_file)
        cols = df.columns.to_list()[:4]
        with st.spinner("Getting Data: Hang On..."):
           results, errors = address_quick(df[cols])

    except:
        st.header('Make Sure File is Loaded First and then hit "Addresses"')

else:
    results, errors = address_quick(address)

m = folium.Map(location=[39.50, -98.35],  zoom_start=3)


with col1:
    st.title('Addresses')
    map_results(results)
    st_folium(m, height=500, width=500)

with col2:
    st.title('Results')
    results.index = np.arange(1, len(results) + 1)
    st.dataframe(results)
    csv = convert_df(results)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Results.csv',
        mime='text/csv')
    try:
        if errors.shape[0] > 0:

            st.header('Errors')
            errors.index = np.arange(1, len(errors) + 1)
            st.dataframe(errors)
            # st.table(errors.assign(hack='').set_index('hack'))
            csv2 = convert_df(errors)
            st.download_button(
                label="Download Errors as CSV",
                data=csv2,
                file_name='Errors.csv',
                mime='text/csv')
    except:
        pass

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
