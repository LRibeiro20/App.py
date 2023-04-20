from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.response_selection import get_random_response

bot = ChatBot(
    "Luis",  
    response_selection_method=get_random_response,
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///database.sqlite3',

    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'Sorry, I am not sure how to respond to that.',
            'maximum_similarity_threshold': 0.4
           
        } 
    ]
)

trainer = ChatterBotCorpusTrainer(bot)
trainer.train("chatterbot.corpus.custom") 






def get_response(user_input):
    return bot.get_response(user_input)


   


# while True:
#     try:
#         request = bot.get_response(input("user: "))
#         print("bot: ", request)

#     except(KeyboardInterrupt, EOFError, SystemExit):
#         break
        