import streamlit as st


sad_page = st.Page("sad.py", title="Checkout your e:MBTI", icon='🌈')
explanation_page = st.Page("explanation.py", title="About e:MBTI", icon='❓')
pg = st.navigation(
                    {"Test": [sad_page],
                    "Explanation": [explanation_page]}
                    )
pg.run()