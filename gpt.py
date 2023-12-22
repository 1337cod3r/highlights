from openai import OpenAI

api_key = 'APIKEY'

client = OpenAI(
    # This is the default and can be omitted
    api_key=api_key,
)

def completions(prompt):
	chat_completion = client.chat.completions.create(
	    messages=[
	        {
	            "role": "user",
	            "content": prompt,
	        }
	    ],
	    model="gpt-3.5-turbo",
	)
	return chat_completion


print(completions("hello"))
