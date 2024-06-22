import streamlit as st
from funcs import create_paraphrase_bart
from funcs import create_paraphrase_T5
from funcs import is_generated_by_ai
from funcs import set_seed



st.set_page_config(
    page_title="AI GEN APP",
    page_icon="📈",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.markdown("""
# AI GENERERATED TEXT RECOGNITION (AGTR)

AGTR provides one usefull service - Understanding if the text was generated by AI. It utilizes HuggingFace transformer model to solve the task. 

## Enter your text and see the magic!

""")


# max_length, num_return_sequences, early_stopping - добавляем на них крутилки
full_text = """ 
            “Wings of Fire: An Autobiography of Abdul Kalam”  is written by Dr. APJ Abdul Kalam and Arun Tiwari. This book is an autobiography of Dr.
            APJ Abdul Kalam, former president of India. This story sheds light on the journey of a young boy from Rameswaram to become a renowned scientist.
            It reflects how simple living, dedication, strong will and hard work led to success. It also shows how cultural unity impacts the lives of individuals.
            """


set_seed(42)

T5_paraphrased = create_paraphrase_T5(full_text, max_length=80, num_return_sequences=1, early_stopping=True)
print("\nT5 paraphrased text: \n")
st.write(f'**{T5_paraphrased}**')

bart_paraphrased = create_paraphrase_bart(full_text, max_length=100, num_return_sequences=1, early_stopping=True)
print("\nBart paraphrased text: \n")
print(bart_paraphrased)




user_input = st.text_area('Enter text','', placeholder='Этот текст будет использован для проверки')
output = is_generated_by_ai(user_input)

st.write("Paraphrased Text: ")
st.write(f'**{output}**')