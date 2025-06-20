
#将数据/hujinwu/code/robotchat/Firefly/raw_data/sample1k_qwen.json
#格式转换为
#模版格式{"conversation_id":1, "category":"Brainstorming", "conversation": [{"human":"xxx", "assistant":"xxxx"},{"human":"xxx", "assistant":"xxxx"}]}


import json
import os
import random
import argparse

chatbot_system = '''
You are a person who is good at chatting. Now you are communicating with someone, but you don't know their identity. You need to actively learn about his background information, intentions, and needs during the conversation and then follow the topic that he is interested in.
You only talk about one short sentence each time.
During the conversation, you should strive to maintain appropriate output as much as possible and avoid generating repetitive content. Additionally, refrain from generating phrases like "How can I help you?" and instead, proactively propose topics relevant to the user's input to stimulate their interest in chatting.
During the conversation, please refrain from focusing the discussion on yourself, but rather pay more attention to topics relevant to the user's background.
DO NOT always ask questions.
'''

human1 = 'Your first response is:'
def convert_data_format_cl(file_path1,outputs_path, alpha = 3, beta = 2):
    
    # # 原始值
    # alpha = 3
    # beta = 2
    
    with open(file_path1, 'r') as f:
        datas = json.load(f)
        new_datas = []
        i = 1
        accept_train = 0
        for data in datas:
            new_data = {}
            new_data['conversation_id'] = i
            i = i+1
            new_data['category'] = data['name']
            new_data['system'] = chatbot_system
            human_content = [human1]
            assistant_content = []
            flag = 1
            diff = 0
            for conversation in data['conversation']:
                if conversation['role'] == 'assistant':
                    assistant_content.append(conversation['content'])
                    if conversation['reject'] != None:
                        for j in range(3):
                            # if conversation['accept_score'][j] <= (alpha - 1) or conversation['accept_score'][j]-conversation['reject_score'][j] < 0:   # 这是原来跑实验用的
                            if conversation['accept_score'][j] <= (alpha - 1):     # 目前是严格遵循论文的格式，也有效果，就用这个
                                flag = 0
                                break
                            elif conversation['accept_score'][j]-conversation['reject_score'][j] > 0:
                                diff = diff + 1
                        if diff < beta:
                            flag = 0
                elif conversation['role'] == 'user':
                    human_content.append(conversation['content'])
            new_conversations = []
            for human, assistant in zip(human_content, assistant_content):
                new_conversations.append({'human': human, 'assistant': assistant})
            new_data['conversation'] = new_conversations
            # print(new_data)
            if flag == 1:
                new_datas.append(new_data)
                accept_train = accept_train + 1
        print(accept_train)
            # new_datas.append(new_data)
        print(len(new_datas))

        os.makedirs(os.path.dirname(outputs_path), exist_ok=True)

        with open(outputs_path, 'w', encoding='utf-8') as output_file:
            for data in new_datas:
                json.dump(data, output_file)
                output_file.write('\n')
                print(data)


def convert_data_format_origin(file_path1,outputs_path):
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
        os.makedirs(os.path.dirname(outputs_path), exist_ok=True)
        with open(outputs_path, 'w', encoding='utf-8') as output_file:
            for data in new_datas:
                json.dump(data, output_file)
                output_file.write('\n')
                print(data)



if __name__ == '__main__':
    # file1 = "/hujinwu/wyf/projects/robotchat/results/qwen32b_cl/qwen32b_cl_train_s04.json"
    
    # output_file = "/hujinwu/code/robotchat/Firefly/data/qwen32b_cl/qwen32b_cl_train_s04_a_4.jsonl"
    # convert_data_format_cl(file1, output_file, alpha=4)

    # output_file = "/hujinwu/code/robotchat/Firefly/data/qwen32b_cl/qwen32b_cl_train_s04_b_1.jsonl"
    # convert_data_format_cl(file1, output_file, beta=1)

    # output_file = "/hujinwu/code/robotchat/Firefly/data/qwen32b_cl/qwen32b_cl_train_s04_b_3.jsonl"
    # convert_data_format_cl(file1, output_file, beta=3)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--origin_json", type=str, default='') 
    parser.add_argument("--output_jsonl", type=str, default= '') 
    parser.add_argument("--use_cl", action="store_true")

    args = parser.parse_args()
    
    if args.use_cl:
        convert_data_format_cl(args.origin_json, args.output_jsonl)
        print('use cl')
    else:
        convert_data_format_origin(args.origin_json, args.output_jsonl)
