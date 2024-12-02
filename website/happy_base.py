import streamlit as st


happy_page = st.Page("happy.py", title="Checkout your e:MBTI", icon='🌈')
explanation_page = st.Page("explanation.py", title="About e:MBTI", icon='❓')
pg = st.navigation(
                    {"Test": [happy_page],
                    "Explanation": [explanation_page]}
                    )
pg.run()