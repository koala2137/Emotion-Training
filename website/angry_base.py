import streamlit as st


angry_page = st.Page("angry.py", title="Checkout your e:MBTI", icon='🌈')
explanation_page = st.Page("explanation.py", title="About e:MBTI", icon='❓')
pg = st.navigation(
                    {"Test": [angry_page],
                    "Explanation": [explanation_page]}
                    )
pg.run()