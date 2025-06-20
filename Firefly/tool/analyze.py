import json


chatbot_system = '''
You are a person who is good at chatting. Now you are communicating with someone, but you don't know their identity. You need to actively learn about his background information, intentions, and needs during the conversation and then follow the topic that he is interested in.
You only talk about one short sentence each time.
During the conversation, you should strive to maintain appropriate output as much as possible and avoid generating repetitive content. Additionally, refrain from generating phrases like "How can I help you?" and instead, proactively propose topics relevant to the user's input to stimulate their interest in chatting.
During the conversation, please refrain from focusing the discussion on yourself, but rather pay more attention to topics relevant to the user's background.
DO NOT always ask questions.
'''

human1 = 'Your first response is:'

def count_elements(arr):
    counts = {}  # 创建一个空字典来存储每个元素的计数
    # 遍历数组中的每个元素
    for element in arr:
        # 如果字典中已经存在这个元素，则将它的计数加一
        if element in counts:
            counts[element] += 1
        # 否则，在字典中添加这个元素，并将它的计数设为 1
        else:
            counts[element] = 1

    return counts

def convert_data_format(file_path1):
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
                            if conversation['accept_score'][j] <= 2 or conversation['accept_score'][j]-conversation['reject_score'][j] < 0:
                                flag = 0
                                break
                            elif conversation['accept_score'][j]-conversation['reject_score'][j] > 0:
                                diff = diff + 1
                        if diff < 2:
                            flag = 0
                elif conversation['role'] == 'user':
                    human_content.append(conversation['content'])
            new_conversations = []
            for human, assistant in zip(human_content, assistant_content):
                new_conversations.append({'human': human, 'assistant': assistant})
            new_data['conversation'] = new_conversations
            if flag == 1:
                new_datas.append(new_data)
                accept_train = accept_train+1
        print(accept_train)
        return new_datas

if __name__ == '__main__':
    file1 = "/hujinwu/wyf/projects/robotchat/results/qwen_sft_train_s01.json"
    convert_data_format(file1)