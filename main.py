import streamlit as st
from funcs import create_paraphrase_bart
from funcs import create_paraphrase_T5
from funcs import is_generated_by_ai
from funcs import set_seed, paraphrase, create_paraphrase
from funcs import russian_paraphrase


set_seed(42)

st.set_page_config(
    page_title="AI GEN APP",
    page_icon="📈",
    layout='wide',
)

st.markdown("""
# AI GENERERATED TEXT RECOGNITION (AGTR)

#### Project made by AnomalyDetection Team
            
AGTR provides one usefull service - Understanding if the text was generated by AI. 
It utilizes HuggingFace transformer model to solve the task. 

""")


# max_length, num_return_sequences, early_stopping - добавляем на них крутилки
full_text =  st.text_area('**Исходный текст**','', placeholder = "Может быть запрос на русском или английском языке..")

option = st.selectbox(
    "**Какую модель Вы хотите использовать?**",
    ("T5", "BART", "PARROT", "ENGLISH LANGUAGE", "RUSSIAN LANGUAGE"),
    help="Выбор модели повлияет на перефразировку текста",
    index = None)
if option!=None:
    st.write("Вы выбрали:", option)

    st.write("**Перефразированный текст**")
    if option == "T5":
        with st.spinner("Please wait..."):
            paraphrased = create_paraphrase_T5(full_text, max_length=80, num_return_sequences=1, early_stopping=True)
        st.write(paraphrased)

    elif option == 'BART':
        with st.spinner("Please wait..."):
            paraphrased = create_paraphrase_bart(full_text, max_length=100, num_return_sequences=1, early_stopping=True)
        st.write(paraphrased)
    
    elif option == 'PARROT': #сбоит
        with st.spinner("Please wait..."):
            t5_paraphrased, paraphrased = create_paraphrase(full_text)
        st.write(paraphrased)

    elif option == 'ENGLISH LANGUAGE':
        with st.spinner("Please wait..."):
            paraphrased = paraphrase(full_text)
        st.write(paraphrased)

    elif option == 'RUSSIAN LANGUAGE':
        with st.spinner("Please wait..."):
            paraphrased = russian_paraphrase(full_text)
        st.write(paraphrased)

    else:
        print(None)
    
    from masks import modify_syntax, modify_sentence_length, modify_word_order
    from masks import activate_passive_sentences, add_emotional_elements, change_style_to_conversational
    from masks import scramble_words_with_control, find_synonyms, replace_synonyms


    st.write('**Маскировка**')
    with st.spinner("Please wait..."):
        masked = modify_syntax(paraphrased)
        masked = modify_sentence_length(masked)
        masked = modify_word_order(masked)

        masked = activate_passive_sentences(masked)
        masked = add_emotional_elements(masked)
        masked = change_style_to_conversational(masked)

        # masked = scramble_words_with_control(paraphrased)
        # masked = find_synonyms(paraphrased)
        # masked = replace_synonyms(paraphrased)



    st.write('**Степень хуманизации**')
    st.write("насколько human-like/ai-like получился текст. Чем ближе к 1, тем больше Human-like")
    output = is_generated_by_ai(masked)
    st.write(f'**{output}**')