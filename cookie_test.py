
import streamlit as st

from streamlit_cookies_controller import CookieController

if 'count' not in st.session_state:
    st.session_state['count']=0

client=CookieController()

client.set('hello','Value')

all=client.getAll()

if not all:
    print(all)    
    print(st.session_state.count)    
    st.session_state['count']+=1

print(all)
st.write(all)