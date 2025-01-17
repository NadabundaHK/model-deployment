import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

def main():
    menu = ['Home', 'Machine Learning']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
    elif choice == "Machine Learning":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()


if __name__ == '__main__':
    main()
