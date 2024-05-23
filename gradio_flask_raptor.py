import gradio as gr
import requests

def call_api(url, data=None):
    try:
        if data:
            response = requests.post(url, json=data)
        else:
            response  = requests.post(url)
        

        print(f"response {response}")
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Return the JSON response
            return response.json()
        else:
            # If the request was not successful, print the error status code
            print("Error:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        # If an error occurred during the request, print the error
        print("Error:", e)
        return None
    

url = "http://127.0.0.1:8000/raptor"

def process_question(question):
    # Call the API
    responses = call_api(url  ,{"query" :question})
    # Return the two responses
    return responses['response1'], responses['response2']

# Create the Gradio interface
iface = gr.Interface(
    fn=process_question, 
    inputs=gr.Textbox(label = "Question"), 
    outputs=[gr.Textbox(label = "Vector Query Engine"), gr.Textbox(label = "Raptor Query Engine")],
    live=False,  # Set to False so that it only updates on button click
    title="Raptor RAG Response Demo",
    description="Enter a question, and get responses from Vector Query Engine and Raptor query engine."
)

# Launch the interface
iface.launch(share=True)
