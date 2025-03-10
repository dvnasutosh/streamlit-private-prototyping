from datetime import datetime, timedelta
from time import sleep
from typing import Literal
import streamlit as st
from streamlit_cookies_controller import CookieController
from streamlit_local_storage import LocalStorage

# Initialize session state variable for count if not present
if 'count' not in st.session_state:
    st.session_state['count'] = 0

st.header('Cookie and Storage Testing')

# Create two tabs for storage and cookie testing
storage_tab, cookie_tab = st.tabs(['Storage Tester', 'Cookie Tester'])

def storage_get():
    local_client = LocalStorage()
    all_items = local_client.getAll()
    if not all_items:
        sleep(1)
    print(all_items)
    return all_items

def cookie_get():
    cookie_client = CookieController()
    all_items = cookie_client.getAll()
    if not all_items:
        print(st.session_state['count'])
        st.session_state['count'] += 1
        sleep(1)
    print(all_items)
    return all_items

def cookie_set(key: str, value: str, expiry: int, unit: Literal['Hours', 'Days']):
    cookie_client = CookieController()
    # Calculate expiry delta based on unit selected
    delta = timedelta(days=expiry) if unit == 'Days' else timedelta(hours=expiry)
    cookie_client.set(key, value=value, expires=datetime.now() + delta)
    cookie_stored = cookie_client.get(key)
    while not cookie_stored:
        print(cookie_stored)
        sleep(1)
    st.success('Cookie set successfully!')

def storage_set(key: str, value: str):
    storage_client = LocalStorage()
    storage_client.setItem(key, value)
    if not storage_client.getItem(key):
        sleep(1)
    st.success('Storage set successfully!')

# Cookie tester tab
with cookie_tab:
    with st.form('Cookie Form'):
        key = st.text_input('Key')
        value = st.text_input('Value')
        expiry_col, unit_col = st.columns([0.7, 0.3])
        expiry = expiry_col.number_input("Expiry time", step=1, min_value=0, value=1)
        unit = unit_col.selectbox('Select Unit', options=['Hours', 'Days'])
        submitted = st.form_submit_button('Set Cookie')
        if submitted:
            cookie_set(key, value, expiry, unit)
    st.write("Current Cookies:")
    st.write(cookie_get())

# Storage tester tab
with storage_tab:
    with st.form('Storage Form'):
        key = st.text_input('Key')
        value = st.text_input('Value')
        submitted = st.form_submit_button('Set Storage')
        if submitted:
            storage_set(key, value)
    st.write("Current Storage Items:")
    st.write(storage_get())
