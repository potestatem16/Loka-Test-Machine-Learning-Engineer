import streamlit as st
import replicate
import os
import torch
import pandas as pd
import numpy as np
import transformers
from sentence_transformers import SentenceTransformer, util
from PIL import Image
os.getcwd()

os.chdir(r"G:\My Drive\BERT Modelo de Lenguaje\Loka")

  # Cargar los embeddings
document_embeddings = np.load('document_embeddings.npy')

df = pd.read_csv("AWS sagemaker documentation.csv")


from transformers import PreTrainedTokenizerFast
# Asumiendo que estás utilizando un modelo BERT, puedes ajustar el tokenizador según tu modelo específico
tokenizer = PreTrainedTokenizerFast.from_pretrained("bert-base-uncased")

def truncate_to_token_limit(text, token_limit=1800):
    tokens = tokenizer.tokenize(text)
    truncated_tokens = tokens[:token_limit]
    return tokenizer.convert_tokens_to_string(truncated_tokens)


# Función para buscar las incrustaciones más cercanas
def search(query, top_k=2):
    query_embedding = generate_embeddings([query])[0]
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    results_text = f"\nTop {top_k} results, but include also results to the best of your knowledge, consistent with the following documentation. Provide the documentation titles of interest:\n"
    
    filenames = []  # Lista para almacenar los nombres de los archivos

    for score, idx in zip(top_results[0], top_results[1]):
        idx = idx.item()  # Convierte de tensor a entero
        filename = df.iloc[idx]['filename']
        filenames.append(filename)  # Añade el nombre del archivo a la lista
        results_text += filename + "\n"
        document_content = df.iloc[idx]['transformed_JSON_dir content']
        truncated_content = truncate_to_token_limit(document_content, 1800)
        results_text += truncated_content + "\n\n"

    return results_text, filenames  # Devuelve los nombres de los archivos junto con el texto de los resultados


# Sentence Transformer
model = SentenceTransformer('all-mpnet-base-v2')#.to(device)

# Función para generar incrustaciones utilizando el dispositivo
def generate_embeddings(texts):
    # Codificar los textos y mover las incrustaciones al dispositivo
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings#.to(device)

# question = "What is SageMaker?"
# search_result = search(question)
# promt = "Question: " + question + "\n\n" + search_result

# os.environ['REPLICATE_API_TOKEN'] = 'r8_Zg1v0cRZwbVI8WKCl5KP0QXV1efe3or4g7yKv'
os.environ['REPLICATE_API_TOKEN'] = 'r8_ZoboQCK8ShD2AE31PSqZtmtvzTjujbw0qaD1T'


api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

# output = api.run(
#   "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
#   input={
#     "debug": False,
#     "top_k": 50,
#     "top_p": 1,
#     "prompt": promt,
#     "temperature": 0.5,
#     "system_prompt": "You are a assitant focus on helping programmers about AWS Sagemaker Documentation, and all related questions",
#     "max_new_tokens": 500,
#     "min_new_tokens": -1
#   }
# )
# print(output)


# out = []
# for value in output:
#     out.append(value)
    
# poem = ''.join(out)

# # Imprime el poema
# print(poem)


# App title
st.set_page_config(page_title="🦙💬 Llama2/BERT AWS Sagemaker Assistant Assistant")

# Replicate Credentials
with st.sidebar:
    
    image = Image.open('AWS.png')
    
    # # Muestra la imagen en el sidebar
    # st.image(image, caption='AWS_sagemaker', use_column_width=True)
    
    # Muestra la imagen en el sidebar con un ancho de 100 píxeles
    st.image(image, width=300)
    
    
    st.title('🦙💬 Llama2/BERT AWS Sagemaker Assistant Assistant \n Loka Test from Manuel Diaz')
    

    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
# def generate_llama2_response(prompt_input):
#     string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
#     for dict_message in st.session_state.messages:
#         if dict_message["role"] == "user":
#             string_dialogue += "User: " + dict_message["content"] + "\n\n"
#         else:
#             string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
#     output = replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
#                            input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
#                                   "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})
#     return output


def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    
    # Agregar la búsqueda de contexto a la entrada del prompt
    search_result, filenames = search(prompt_input)  # Obtiene los nombres de los archivos de la función search
    prompt_input = "Question: " + prompt_input + "\n\n" + search_result

    output = api.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={
            "debug": False,
            "top_k": 50,
            "top_p": top_p,
            "prompt": prompt_input,
            "temperature": temperature,
            "system_prompt": "You are a assitant focus on helping programmers about AWS Sagemaker Documentation, and all related questions",
            "max_new_tokens": 500,
            "min_new_tokens": -1
            }
        )

    return output, filenames
    # replicate.run('a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5', 
    #                        input={"prompt": f"{string_dialogue} {prompt_input} Assistant: ",
    #                               "temperature":temperature, "top_p":top_p, "max_length":max_length, "repetition_penalty":1})

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, filenames = generate_llama2_response(prompt)  # Obtiene los nombres de los archivos de la función generate_llama2_response
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            # Añade los nombres de los archivos como "fuentes" al final de la respuesta
            full_response += "\n" + "\nSources:\n" + "\n".join("* " + filename for filename in filenames)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             response = generate_llama2_response(prompt)
#             placeholder = st.empty()
#             full_response = ''
#             for item in response:
#                 full_response += item
#                 placeholder.markdown(full_response)
#             placeholder.markdown(full_response)
#     message = {"role": "assistant", "content": full_response}
#     st.session_state.messages.append(message)