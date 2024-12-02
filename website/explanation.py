import streamlit as st

st.markdown("""
            <style>
            body {
                background-color: beige;
            }
            h2 {
                font-size: 20px;
            }
            .cont_title {
                font-weight:bold;
                font-size: 18px;
            }
            </style>
            """, unsafe_allow_html=True)

st.title("What is e:MBTI?")
st.markdown("---")
first = st.container(border=True)
second = st.container(border=True)
third = st.container(border=True)

with st.expander("🤔 왜 시작했나요?"):
    st.write('''
        📌 표정은 우리에게 많은 것을 말해줍니다. 특히 그 사람의 감정까지도요. 하지만 같은 감정이 항상 같은 표정으로 드러나는 것은 아닙니다. 
        기쁨도, 슬픔도, 분노도 사람마다 각기 다른 표정으로 나타납니다. 타인은 나의 표정에서 어떤 감정을 읽어낼까요? 또 나의 표정은 타인에게 어떤 감정으로 읽힐까요?
        ''')
    st.write('''
        📌 저마다의 성향을 분류하는 MBTI처럼, 저희는 감정-표정 유형을 검사합니다. 팀원들과 직접 데이터를 만들어가며 학습시킨 표정 인식 모델로 여러분의 e:MBTI를 확인해보세요.
        ''')
with st.expander("🤓 분류 기준이 뭔가요?"):
    st.write('''
        📌 아무런 표정도 짓지 않았는데 오해받으신 적 없으신가요? 
        여러분의 평상시 표정에서 '중립(neutral)'을 제외하고 가장 높은 수치를 보이는 감정을 분석합니다. 
        늘 웃는 상인 사람(happy), 괜히 챙겨주고 싶은 사람(sad), 왠지 열정 가득한 사람(angry).
        다른 사람들이 당신을 그렇게 보지는 않던가요?
        ''')
    st.write('''
        📌 지금 서 계신 모니터에는 비슷한 결과를 받은 팀원들의 데이터로 학습시킨 모델이 연결되어 있습니다. 
        꾸며내거나 과장하지 않은 표정 데이터가 여러분을 더 잘 읽어내길 바라며, 당신의 다양한 표정들에서 분석된 감정의 수치가 변화하는 정도를 계산합니다.
        당신은 얼마나 열려있나요?(open / closed) 
        ''')
with st.expander("🤗 메시지가 무엇인가요?"):
    st.write('''
        📌 MBTI가 유행하기 전까지 현실적인 조언을 바라는 T와 감정적인 위로를 바라는 F는 서로를 이해할 수 없었을지 모릅니다.
        저희는 자신과 다를 수 있다는 사실을 인식하는 것만으로 더 많은 것을 보고, 느끼고, 포용할 수 있다고 생각합니다.
        e:MBTI가 그 작은 기회를 제공할 수 있다면 좋겠습니다.
        ''')
    
# first.subheader("🤔 왜 시작했나요?")
# first.markdown
# first.write()
# second.subheader("🤓 어떻게 검사하나요?")
# third.subheader("🤗 메시지가 무엇인가요?")

# col1, col2, col3 = st.columns(3)
# left = col1.container(border=True, height=400)
# middle = col2.container(border=True, height=400)
# right = col3.container(border=True, height=400)

# left.markdown("<span class='cont_title'>🤔 왜 시작했나요?</span>", unsafe_allow_html=True)
# # left.markdown("---")
# left.write('  표정으로 인한 오해를 겪으신 적 없으신가요? 표정은 우리에게 많은 것을 말해줍니다. 특히 그 사람의 감정까지도요. 하지만 같은 감정이 항상 같은 표정으로 드러나는 것은 아닙니다. 기쁨도, 슬픔도, 분노도 모두 저마다의 방식으로 나타나죠.')
# middle.markdown("<span class='cont_title'>🤓 어떻게 검사하나요?</span>", unsafe_allow_html=True)
# # middle.markdown("---")
# right.markdown("<span class='cont_title'>🤗 메시지가 무엇인가요?</span>", unsafe_allow_html=True)
# # right.markdown("---")

    
