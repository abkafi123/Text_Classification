import tensorflow as tf
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
import streamlit as st
from transformers import TextClassificationPipeline
import matplotlib.pyplot as plt

#import part ended

#tokinizer and model are init
def analysis(string:str)->dict: 
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = TFDistilBertForSequenceClassification.from_pretrained("./model", num_labels=140,problem_type="multi_label_classification")

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)
    x = pipe(string)
    label1 = x[0]

    return {
    'label':label1['label'],
    'score':label1['score']
    }


#from here the work of the interface startes

#side part of the interface 
csvfile = st.sidebar.file_uploader(label='Upload a CSV file', type=['csv'])

original_title = '<p style="color:#EEE4AB; font-size: 14px;">Loading from file is under Construction</p>'
st.sidebar.markdown(original_title, unsafe_allow_html=True)


st.sidebar.write('The model used in this app is trained on **WOS** dataset to classify Research Papers using Abstracts')


#middle part of the interface 

st.write('''
Research paper Category Analysis
''')

text = st.empty()

input_text = text.text_area("Enter the Abstract",key=1)

if len(input_text) != 0:
    output = analysis(input_text)
    st.write(f'The Category: {output["label"]}')
    st.write(f'The confidence {(output["score"]*100):.2f}')
    text.text_area("Enter the Abstract",value = '',key=2)
else:
    st.warning("The Text Abstract Can't be empty")



# middle part end