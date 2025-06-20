
#将数据/hujinwu/code/robotchat/Firefly/raw_data/sample1k_qwen.json
#格式转换为
#模版格式{"conversation_id":1, "category":"Brainstorming", "conversation": [{"human":"xxx", "assistant":"xxxx"},{"human":"xxx", "assistant":"xxxx"}]}


import json
import os
import random

chatbot_system = '''
You are a person who is good at chatting. Now you are communicating with someone, but you don't know their identity. You need to actively learn about his background information, intentions, and needs during the conversation and then follow the topic that he is interested in.
You only talk about one short sentence each time.
During the conversation, you should strive to maintain appropriate output as much as possible and avoid generating repetitive content. Additionally, refrain from generating phrases like "How can I help you?" and instead, proactively propose topics relevant to the user's input to stimulate their interest in chatting.
During the conversation, please refrain from focusing the discussion on yourself, but rather pay more attention to topics relevant to the user's background.
DO NOT always ask questions.
'''

human1 = 'Your first response is:'
def convert_data_format(file_path1,outputs_path):
    with open(file_path1, 'r') as f:
        datas = json.load(f)
        new_datas = []
        i = 1
        for data in datas:
            new_data = {}
            new_data['conversation_id'] = i
            i = i+1
            new_data['category'] = data['name']
            new_data['system'] = chatbot_system
            human_content = [human1]
            assistant_content = []
            for conversation in data['conversation']:
                if conversation['role'] == 'assistant':
                    assistant_content.append(conversation['content'])
                elif conversation['role'] == 'user':
                    human_content.append(conversation['content'])
            new_conversations = []
            for human, assistant in zip(human_content, assistant_content):
                new_conversations.append({'human': human, 'assistant': assistant})
            new_data['conversation'] = new_conversations
            # print(new_data)
            new_datas.append(new_data)
        # print(len(new_datas))
        with open(outputs_path, 'w', encoding='utf-8') as output_file:
            for data in new_datas:
                json.dump(data, output_file)
                output_file.write('\n')
                print(data)

if __name__ == '__main__':
    file1 = "/hujinwu/wyf/projects/robotchat/results/qwen1.5_72b_allsft/qwen_sft_train_all_s00.json"
    output_file = "/hujinwu/code/robotchat/Firefly/data/qwen_sft_all_data/qwen_sft_train_all_s00.jsonl"
    convert_data_format(file1, output_file)
