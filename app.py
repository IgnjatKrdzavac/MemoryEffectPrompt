
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
import apikey

os.environ["OPENAI_API_KEY"] = apikey.apikey

llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=512)

conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)

conversation_history = []

def chatbot(pt):
    res = conversation.predict(input=pt)
    return res

if __name__=='__main__':
    while True:
        print('########################################\n')
        pt = input('ASK: ')
        if pt.lower()=='end':
            break
        response = chatbot(pt)


        conversation_history.append(("User: " + pt, "ChatGPT: " + response))

        print('\n----------------------------------------\n')
        print('ChatGPT says: \n')
        print(response, '\n')
        
    print("\nConversation History:")
    for idx, (user, bot) in enumerate(conversation_history, start=1):
        print(f"{idx}. {user}\n   {bot}\n")