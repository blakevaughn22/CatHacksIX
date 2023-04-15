import openai

# Set up the OpenAI API client
openai.api_key = "API"

# Set up the model and prompt
model_engine = "gpt-3.5-turbo" 
prompt = "Hello, how are you today?"

planet = input("Input what planet/moon you want fun facts about:")

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    max_tokens=1024,
    messages=[
        {"role": "system", "content": "Output 3 fun facts about the planet/moon that is inputted by the user, if the input is just \"Moon\", assume it is earth's moon. Make sure your output only contains the fun facts"},
        {"role": "user", "content": planet},
    ])

message = response.choices[0]['message']
print("Fun Facts!\n {}".format(message['content']))
