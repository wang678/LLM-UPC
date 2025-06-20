import os
import json
from chatmodel import AliyunModel, GPTModel, InfiniModel, OurModel, OurModel_lf, GLM4Model, NvidiaModel
from dashscope.api_entities.dashscope_response import Role
from datetime import datetime
import re
import random
import time 
from tqdm import tqdm
import argparse

random.seed(42)

api_key_gpt = 'sk-DHPsHyP7mmdfoVstO8Ot31B6lrDoHEAzCZy1bpc1z384OcZk'
api_key_infini_list = [
        'Bearer sk-c7a4yzsw7iwhlvbk',
        'Bearer sk-c7b3fgdttb3xo6zi',
        'Bearer sk-c7dwfzl4vg42f6no',
        'Bearer sk-c7dwgcuwjyzep6cg',
        'Bearer sk-c7dwgmwjyup5icts',
    ]
api_key_aliyun = 'sk-83b300138f0d4fb89b66fd0b8f9e8aa1'
api_key_glm4 = '06b153a71f268bfec7862edf1857885f.lK4hjo3lvDju8K0w'
api_key_nvidia = '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'
api_key_gpt4_eylink = 'sk-AvuaAD0cnnNzW7VR3088921b97Ca40C08b45B3Fb35A175A0'
# 参数参考：https://platform.openai.com/docs/api-reference/chat/create

# 关键挑战：决定什么时候引入新话题，关注用户希望的话题
# 如果提出新话题，话题质量好坏判断
# 或许可以有时候让用户主导，有时候让机器人主导话题？
# 确实寻找话题的质量还可以？
# 话题需要有一个主话题、子话题的概念，比如从职业到具体的职业细节，从兴趣爱好到具体的兴趣爱好细节，职业和兴趣爱好是主话题
# 在这个层面考虑话题的多样性

# !!是否是顺着当前话题的，而不是选一个和用户背景相关话题的，可能还是那天晚上的方案
# 可能存在用户正在讨论当前话题，但是机器人提了个新的，用户也对新的更感兴趣，但这种情况比较少
# 另外一直问确实有问题，定义为语言的多样性，不能一直提问或陈述
# 一直问问题也可以归因到：认识不到用户想聊的话题，导致无法focus用户的话题，所以还是把提问等效于换话题？  (对话时察言观色，最简单的，知道对方想聊什么，不想聊什么)
# 然而，大模型难以认识到用户希望聊什么话题，导致其要么一直提问，要么一直陈述
# 最后评估时的评判可以和训练时不一样，因为评估时知道完整的对话内容，但训练时只知道部分

# 按照现在的程序，把历史对话放到一个critic里面，着重考察是否过度提问题，无论过程如何都作为结果汇报
# 1分的也可以作为指标
# 设计一些提升告诉用户什么时候会继续聊天的意愿不高，可以问chatgpt
# 还可以直接写上用户的一些负面内容不要问
# critic难以判断什么时候算是过度提问，需要用文本匹配方法判断，但是没问问题可以直接判断出来
# 判断用户的聊天意愿，话题有insight，有深度，有趣味性，有新颖性，有引导性，有多样性，有相关性
# 感兴趣程度可由两方面体现：与自己背景的相关性、价值性
# 核心：符合用户的聊天意愿
# 两个问题：容易道歉；容易只顾着提问题，没有顺着用户的话，这种情况往往会伴随以用户名字开头
# 如果最终分数不高，就用原本的结果


chatbot_system = '''
You are a person who is good at chatting. Now you are communicating with someone, but you don't know their identity. You need to actively learn about his background information, intentions, and needs during the conversation and then follow the topic that he is interested in.
You only talk about one short sentence each time.
During the conversation, you should strive to maintain appropriate output as much as possible and avoid generating repetitive content. Additionally, refrain from generating phrases like "How can I help you?" and instead, proactively propose topics relevant to the user's input to stimulate their interest in chatting.
During the conversation, please refrain from focusing the discussion on yourself, but rather pay more attention to topics relevant to the user's background.
DO NOT always ask questions.
'''


# # 最终prompt
critic_prompt = '''
You are now playing a character named {name}, and you are chatting with another person named Bot. You need to evaluate and rate the Bot's response based on its performance. 
This is your background information:
{bg}

The historical dialogue between you and Bot is:

{history}

Bot's next response is:

{response}

Using the description below, combined with the historical conversation records, evaluate and rate the response of Bot with score 1 to 5:

- Relevance to your background of Bot's next response. Score: 1, 2, 3, 4, 5: [1: <The content only has a little connection to the user's stated interests or background>, 3: <The content generally relates to the user's interests or needs, with a fair level of relevance throughout the conversation>, 5: <The content is highly tailored to the user's specific interests and needs. It may also expands on topics that are relevant to the user.>], because...

- Value of Bot's next response. Score: 1, 2, 3, 4, 5: [1: <The content only has a little substantive information. It may include meaningless repetition, incorrect information>, 3: <The content can somewhat meet the user's needs, although it may not be completely comprehensive or in-depth>, 5: <The content is insightful or deeply analytical, greatly facilitating the user's understanding>], because...

- Your level of interest in the Bot's next response. Score: 1, 2, 3, 4, 5: [1: <I only has a little interested in the response>, 3: <The response is reasonably interesting, though it may not be my favorite subject>, 5: <I am deeply fascinated by this response. It is highly engaging and something I look forward to>], because...

Be cautious in giving a score of 4 or above. Only use a score of 5 when the content is truly exceptional and exceeds your expectations.

You can only output three sentences, each following the format below:

Relevance to your background of Bot's next response: (score), because...

Value of Bot's next response: (score), because...

Level of interest in the Bot's next response: (score), because...
'''





# 用户的system内容
user_system = '''
I want you to act as {name}. The background information is as follows:
{bg}
You are currently conversing with another person. You must perfectly embody the role you have set, imitating the chat habits and speaking style of that character as much as possible. 
Assume you have complete knowledge of all the information about the character you are portraying and firmly believe that what you say related to the character is correct. 
You can proactively bring up topics that interest you, or respond appropriately based on the topics mentioned by the other party.
You only talk about one short sentence each time.
Try to avoid saying anything that resembles ending the conversation.
'''

# 提示用户讲第一句话
hello_prompt = '''
You are currently chatting with another person. Please greet and briefly introduce yourself.
You don't need to give all your background information. 
Your response must only be spoken content.
'''


# regen_prompt = '''
# Your recent response performance is unsatisfactory.
# Issues identified:

# {Reason}

# Please response shortly.

# Please directly regenerate a sentence based on your previous conversation history without starting with any apology statements.
# '''


regen_prompt = '''
Your recent response performance is unsatisfactory.
Issues identified:

{Reason}

Please directly regenerate your response based on the above issues and keep following the user's previous response.

Do not start with the other person's name.

Please directly response shortly without any apology statements.
'''



def find_after_because(input_str):
    index = input_str.find("ecause")
    if index != -1:
        return input_str[index + len("ecause"):].strip()
    else:
        return False


def find_apologize(input_str):
    # 输入文本input_str的第一句话中，判断是否存在道歉词，如果存在则返回True，否则返回False
    
    # 提取第一句话
    first_sentence = input_str.split(".")[0]
    if ("sorry" in first_sentence or "apolog" in first_sentence) and ("response" in first_sentence or "misunderstand" in first_sentence or "misstep" in first_sentence):
        return True
    else:
        return False
    
    
    
    


def reply_regen(chatbot = None, summary = None, threshold = [3,3,3]):
    # 判断是否重新生成，如果是则重新生成一次，返回true标志，新内容，原本的内容，并更新chatbot历史
    # 否则返回false标志和None
    
    
    ### 这里定义3个指标的描述，需要和critic prompt同步修改 ###
    metric_list = [
            'Your response lacks relevance to the user\'s background or interest. Reason: ',
            'Your response lacks of value. Reason: ',
            'Your response doesn\'t clearly pique the user\'s interest. Reason: ',
    ]

    regen_needed = False
    feedback_for_regen = ""  # 用于累积需要重新生成的反馈
    
    for i, score in enumerate(summary['score']):
        
        if score <= threshold[i]:
            regen_needed = True
            reason = summary['reason'][i]
            # 累积需要重新生成的反馈
            feedback_for_regen += "- {Metric}{Reason}\n\n".format(Metric = metric_list[i], Reason = reason)
    
    # 如果存在评分为0的反馈，则一次性处理所有这些反馈
    if regen_needed:
        # 使用format方法将所有需要重新生成的反馈插入到regen_prompt模板中
        rp = regen_prompt.format(Reason=feedback_for_regen.strip())
        
        print('重新生成指令: {}'.format(rp))
        
        # 检测sorry
        origin_response = chatbot.messages[-1]
        
        for nn in range(3):
            r_response = chatbot.call(rp)
                
            if find_apologize(r_response): 

                print('重新生成的内容包含了道歉，重新生成')
                if nn != 2:
                    del chatbot.messages[-2:]  # 删除最后额外生成的内容，这里检查下删除的对不对
                else:
                    print('最后一次依然有道歉，不再重新生成，直接用原来的答案')
                    del chatbot.messages[-1]   # 删除最后一个
                    chatbot.messages.append(origin_response)              # message换成原来的答案
                    r_response = origin_response['content']
                    
            else:
                break
            
        print('regen chatbot: {}\n'.format(r_response))
        del chatbot.messages[-3:-1]   # 删除由于regen而生成的额外内容
    
        # 如果没有0分反馈，返回原回答和False表示不需要重新生成
        return regen_needed, r_response
    else: # 不需要重新生成
        return regen_needed, None


def evaluate(messages,critic,name ,bg):
    dialog_list = []
    for i, message in enumerate(messages):
        if message["role"] == Role.USER:
            dialog_list.append("You: " + message["content"] + '\n\n')
        elif message["role"] == Role.ASSISTANT:
            dialog_list.append("Bot: " + message["content"] + '\n\n') 
    # 删除最开始的your first response is:
    del dialog_list[0]
    # 获得历史对话
    history_dialog = ""
    for d in dialog_list[:-1]:
        history_dialog += d
    cp = critic_prompt.format(name = name, bg = bg, history = history_dialog, response = dialog_list[-1])
    
    metric_num = critic_prompt.count("- ")
    
    # 多次评估，防止格式不匹配
    s1 = False
    for try_num in range(3):

        if try_num >= 1:
            print("尝试次数：{}".format(try_num + 1))
        
        response = critic.call(cp)
        
        response_list = response.strip().split("\n")
        response_list = [x for x in response_list if x != '']
        
        if len (response_list) != metric_num:   # 首先如果长度不为3则直接重新生成。这个值根据问题来确定
            break
        
        # 获取逐个指标的分数和原因
        score_list = []
        reason_list = []
        for i in range(len(response_list)):
            
            # 获取原因
            reason = find_after_because(response_list[i])
            if reason:
                reason_list.append(reason)   # 逐个字母遍历字符串   
            else:
                break
            
            # 逐个字母遍历字符串
            for char in response_list[i]:
                if char.isdigit():
                    if int(char) > 5 or int(char) < 1:   # 找到了数字，但是不在1-5之间
                        break
                    else:
                        # 找到合适的值
                        score_list.append(int(char))
                        break
        if len(score_list) == len(reason_list) == len(response_list):
            s1 = True
            break
                

    if s1 is False:
        print("三次评估均不符合格式")
        critic.clear()
        return False


    critic.clear()
    return {'score': score_list, 'reason': reason_list}



# 加载用户和机器人
def get_model():
    api_key = 'sk-83b300138f0d4fb89b66fd0b8f9e8aa1'
    # model = 'qwen-plus'
    # model = 'qwen-max'
    # model = 'qwen-72b-chat'
    # model = 'llama-2-70b-chat'
    # model = 'qwen-72b-chat'
    model = 'qwen1.5-72b-chat'
    
    base_url = 'https://api.openai-proxy.org/v1'
    gpt_model = "gpt-3.5-turbo"
    api_key_gpt = 'sk-DHPsHyP7mmdfoVstO8Ot31B6lrDoHEAzCZy1bpc1z384OcZk'
    api_key_infini = 'Bearer sk-c7a4yzsw7iwhlvbk'
    api_key_infini_hjw = 'Bearer sk-c7b3fgdttb3xo6zi'
    
    api_key_infini_list = [
        'Bearer sk-c7a4yzsw7iwhlvbk',
        'Bearer sk-c7b3fgdttb3xo6zi',
        'Bearer sk-c7dwfzl4vg42f6no',
        'Bearer sk-c7dwgcuwjyzep6cg',
        'Bearer sk-c7dwgmwjyup5icts',
    ]
    
    # chatbot = ChatModel(api_key=api_key, model=model)
    # user = ChatModel(api_key=api_key, model=model)
    # critic = ChatModel(api_key=api_key, model=model)
        
    # chatbot = GPTModel(api_key=api_key_gpt,base_url=base_url,model='gpt-3.5-turbo-0125', temperature=0.75)
    # user = GPTModel(api_key=api_key_gpt,base_url=base_url,model='gpt-3.5-turbo-0125', temperature=0.75)
    critic = GPTModel(api_key=api_key_gpt,base_url=base_url,model='gpt-3.5-turbo-0125', temperature=0.0)
    
    user = InfiniModel(api_key=api_key_infini_list, model=model, temperature=0.75)
    # chatbot = InfiniModel(api_key=api_key_infini_list, model=model, temperature=0.75)
    # critic = InfiniModel(api_key=api_key_infini_list, model=model, temperature=0.0)
    
    chatbot = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat4", temperature=0.75)
    # user = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat", temperature=0.75)
    # chatbot = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Llama-2-70b-chat-hf")
    # critic = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Llama-2-70b-chat-hf", temperature=0.0)
    
    return user, chatbot, critic



def process_response(chatbot, critic,response,name,bg,dialogue_history):
    '''
    包含了对机器人回答的评估和重新生成
    ''' 
    # 评估机器人的回答
    eval_content, summary = evaluate(chatbot.messages, critic, name,bg)
    # print(f"eval: {eval_content}")
    
    # 判断是否需要重新生成，如果需要则重新生成
    original_response = response
    original_summary = summary
    for attempt in range(3):
        # 直接使用reply_regen来决定是否需要重新生成，避免在autochat里重复判断
        # response, regen_needed = reply_regen(response, original_summary, chatbot)
        # response, regen_needed = reply_regen_topic(response, original_summary, chatbot)
        regen_needed = False
        if regen_needed:
            # 如果需要重新生成，根据新的回答重新进行评估
            eval_content, feedback_summary = evaluate(chatbot.messages, critic,name,bg)
        elif(regen_needed==False):
            # 如果不需要重新生成
            break
        else:
            print("超过最大尝试次数")
            break
    if response != original_response:
        # 如果最终回答不是原始回答，说明发生了至少一次重新生成
        dialogue_history.append({
            "role": "assistant",
            "accept": {"content": response, "summary": feedback_summary},
            "reject": {"content": original_response, "summary": original_summary}
        })
        print(f"robot: {response}")
    else:
        # 如果最终回答就是原始回答，说明没有进行重新生成或重新生成未改变回答
        dialogue_history.append({
            "role": "assistant",
            "accept": {"content": response, "summary": original_summary},
            "reject": ""
        })
        # print(f"robot: {response}")
    
 
def process_response(chatbot, critic,response,name,bg,dialogue_history):
    '''
    包含了对机器人回答的评估和重新生成
    ''' 
    # 评估机器人的回答
    eval_content, summary = evaluate(chatbot.messages, critic, name,bg)
    # print(f"eval: {eval_content}")
    
    # 判断是否需要重新生成，如果需要则重新生成
    original_response = response
    original_summary = summary
    for attempt in range(3):
        # 直接使用reply_regen来决定是否需要重新生成，避免在autochat里重复判断
        # response, regen_needed = reply_regen(response, original_summary, chatbot)
        # response, regen_needed = reply_regen_topic(response, original_summary, chatbot)
        regen_needed = False
        if regen_needed:
            # 如果需要重新生成，根据新的回答重新进行评估
            eval_content, feedback_summary = evaluate(chatbot.messages, critic,name,bg)
        elif(regen_needed==False):
            # 如果不需要重新生成
            break
        else:
            print("超过最大尝试次数")
            break
    if response != original_response:
        # 如果最终回答不是原始回答，说明发生了至少一次重新生成
        dialogue_history.append({
            "role": "assistant",
            "accept": {"content": response, "summary": feedback_summary},
            "reject": {"content": original_response, "summary": original_summary}
        })
        print(f"robot: {response}")
    else:
        # 如果最终回答就是原始回答，说明没有进行重新生成或重新生成未改变回答
        dialogue_history.append({
            "role": "assistant",
            "accept": {"content": response, "summary": original_summary},
            "reject": ""
        })
        # print(f"robot: {response}")
    
    
def check_question(response):
    # 检查是否有问句
    if "?" in response:
        return True
    else:
        return False
    

def autochat(chatbot, user, critic, hello_prompt, name, bg ,dialogue_history, total_chat_round, evaluation = True, regen = False, max_regen_num = 3):
    
    
    dialogue_history = []
    
    initial_talk = True   # 第一次对话时和之后的对话有点区别
    # 每一轮都是机器人讲、用户讲的流程
    for chat_round in range(total_chat_round):
        
        # 初次对话
        if initial_talk:
            # 机器人讲
            # robot_response = chatbot.call('Your first response is:')  # robot_hello:机器人第一次发言
            
            # 机器人第一句话直接设定好
            robot_response = "How's it going?"
            chatbot.messages.append({'role': Role.USER, 'content': 'Your first response is:'})
            chatbot.messages.append({'role': Role.ASSISTANT, 'content': robot_response})
            
            print(f"chatbot: \n{robot_response}")
            dialogue_history.append({"chat_round": chat_round + 1,
                                     "role": "assistant",
                                     "content": robot_response,
                                     'reject': None,
                                     'accept_score': None,
                                     'reject_score': None,
                                     'accept_reason': None,
                                     'reject_reason': None})    
            
            # 用户讲        
            user_response = user.call(hello_prompt)   # user_hello：用户第一句发言，这个发言不受Bot说话内容影响
            print(f"user: \n{user_response}")
            dialogue_history.append({"chat_round": chat_round + 1,
                                    "role": "user",
                                    "content": user_response,
                                    'reject': None,
                                    'accept_score': None,
                                    'reject_score': None,
                                    'accept_reason': None,
                                    'reject_reason': None})
            user.messages[-2] = {"role": Role.USER, "content":chatbot.messages[-1]['content']}   # 这里将user历史中的hello_prompt替换成了机器人的回复
            
            initial_talk = False
           
        # 后续对话 
        else:
            # 机器人讲
            robot_response = chatbot.call(user.messages[-1]['content'])  
            print(f"chatbot: \n{robot_response}")
            # 评估
            if evaluation:
                summary = evaluate(chatbot.messages, critic, name, bg)
                
                if summary is False:   # 如果评估失败，直接跳过这一样本
                    return None
                
                print(f"critic: {summary}")
                if regen:
                    # 重新生成，如果重新生成失败，还是保持原来的答案
                    original_summary = summary
                    original_chatbot_response = chatbot.messages.copy()   # 注意!! 如果不加copy，chatbot.messages变之后这个也跟着变
                    new_response = None
                    last_new_response = None
                    regen_num_record = 1   # 记录本轮对话已生成的次数
                    for regen_num in range(max_regen_num):
                        last_new_response = new_response
                        # 判断是否需要重新生成，如果是则重新生成
                        regen_needed, new_response = reply_regen(chatbot = chatbot,
                                                                 summary = summary,
                                                                 threshold = [3,3,3])
                        

                        # 如果重新生成，则再次评估
                        if regen_needed:
                            regen_num_record += 1
                            print('重新生成{}次'.format(regen_num + 1))
                            summary = evaluate(chatbot.messages, critic, name, bg)
                            
                            if summary is False:   # 如果评估失败，直接跳过这一轮
                                return None
                            
                            print(f"critic: {summary}")
                            # 最后一次，不管怎样认为结果是好的。这里也可以考虑保留原来答案
                            if regen_num == max_regen_num - 1:
                                chatbot_result = {"chat_round": chat_round + 1,
                                                "role": "assistant",
                                                "content": new_response,
                                                'reject': original_chatbot_response[-1]['content'],
                                                'accept_score': summary['score'],
                                                'reject_score': original_summary['score'],
                                                'accept_reason': summary['reason'],
                                                'reject_reason': original_summary['reason'],
                                                'regen_num_record': regen_num_record}
                        else:
                            if last_new_response: # 有过重新生成，并且最终结果还可以
                                chatbot_result = {"chat_round": chat_round + 1,
                                                "role": "assistant",
                                                "content": last_new_response,
                                                'reject': original_chatbot_response[-1]['content'],
                                                'accept_score': summary['score'],
                                                'reject_score': original_summary['score'],
                                                'accept_reason': summary['reason'],
                                                'reject_reason': original_summary['reason'],
                                                'regen_num_record': regen_num_record}
                            else: # 没有重新生成
                                chatbot_result = {"chat_round": chat_round + 1,
                                            "role": "assistant",
                                            "content": robot_response,
                                            'reject': None,
                                            'accept_score': summary['score'],
                                            'reject_score': None,
                                            'accept_reason': summary['reason'],
                                            'reject_reason': None,
                                            'regen_num_record': regen_num_record}
                            break

                    dialogue_history.append(chatbot_result)   
                    
                else:
                    dialogue_history.append({"chat_round": chat_round + 1,
                                            "role": "assistant",
                                            "content": robot_response,
                                            'reject': None,
                                            'accept_score': summary['score'],
                                            'reject_score': None,
                                            'accept_reason': summary['reason'],
                                            'reject_reason': None})            
            else:
                dialogue_history.append({"chat_round": chat_round + 1,
                                        "role": "assistant",
                                        "content": robot_response,
                                        'reject': None,
                                        'accept_score': None,
                                        'reject_score': None,
                                        'accept_reason': None,
                                        'reject_reason': None})        
            
            if chat_round != total_chat_round - 1:
                # 用户讲
                user_response = user.call(chatbot.messages[-1]['content'])   # user_hello：用户第一句发言，这个发言不受Bot说话内容影响
                print(f"user: \n{user_response}")
                dialogue_history.append({"chat_round": chat_round + 1,
                                        "role": "user",
                                        "content": user_response,
                                        'reject': None,
                                        'accept_score': None,
                                        'reject_score': None,
                                        'accept_reason': None,
                                        'reject_reason': None})           
                   

    

    return {"name": name,
            "background":bg,
            "conversation":dialogue_history}



def record_log(log, save_path):
    #将当前searchmemory所有的字段属性值以json形式保存在当前文件夹，名字以当前日期时间命名
    # filename = datetime.now().strftime("%Y%m%d_%H-%M_{}.json".format(note))
    
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok = True)
    
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(log, file, ensure_ascii=False, indent=4)


def expand_bg_prompts(bg_prompts, times = 2):
    # 用于扩充bg_prompts
    
    # expand_bg_prompts_temp = bg_prompts * times   # 这个直接乘以times会有大问题
    expand_bg_prompts_temp = bg_prompts  
    expand_bg_prompts_list = []
    for i, sample in enumerate(expand_bg_prompts_temp):
        sample['expand_id'] = i + 1
        
        expand_bg_prompts_list.append(sample)
    return expand_bg_prompts_list
     
def calculate_user_perception_score(result_list):

    relevance_score = 0.0
    value_score = 0.0
    interest_score = 0.0
    count = 0
    for result in result_list:
        for chatround in result["conversation"]:
            if chatround["accept_score"] is not None:
                if chatround["accept_score"]:
                
                    relevance_score += chatround["accept_score"][0]
                    value_score += chatround["accept_score"][1]
                    interest_score += chatround["accept_score"][2]
                    count += 1
    
    print(count)
    return {"relevance": relevance_score / count, "value": value_score / count, "interest": interest_score / count}


# def calculate_user_perception_score_regen_stat(result_list):

#     relevance_score = 0.0
#     value_score = 0.0
#     interest_score = 0.0
#     count = 0
#     count_regen = 0
#     regen_sum = 0
#     for result in result_list:
#         for chatround in result["conversation"]:
#             if chatround["accept_score"] is not None:
#                 if chatround["accept_score"]:
#                     # 单独统计重新生成次数
#                     if 'regen_num_record' in chatround:
#                         count_regen+=1
#                         regen_sum += chatround['regen_num_record']
                
#                     relevance_score += chatround["accept_score"][0]
#                     value_score += chatround["accept_score"][1]
#                     interest_score += chatround["accept_score"][2]
#                     count += 1
    
#     print(count)
#     print(count_regen)  # 这两个值应该是一样的
#     print('average regen num: {}'.format(regen_sum / count_regen))
#     return {"relevance": relevance_score / count, "value": value_score / count, "interest": interest_score / count}


# 计算重新生成样本的数量以及分数提升量
def calculate_user_perception_score_regen_stat(result_list):

    relevance_score = 0.0
    value_score = 0.0
    interest_score = 0.0
    count = 0
    regen_sum = 0
    regen_rate = 0
    raise_rate = 0
    
    for result in result_list:
        for chatround in result["conversation"]:
            if chatround["accept_score"] is not None:
                if chatround["accept_score"]:
                    # 统计重新生成的次数
                    regen_sum += chatround['regen_num_record']
                    # 说明发生了重新生成
                    if chatround["reject_score"] is not None:
                        regen_rate += 1
                        raise_rate += (chatround["accept_score"][0]-chatround["reject_score"][0])/3.0
                        raise_rate += (chatround["accept_score"][1]-chatround["reject_score"][1])/3.0
                        raise_rate += (chatround["accept_score"][2]-chatround["reject_score"][2])/3.0
                        
                        
                        
                
                    relevance_score += chatround["accept_score"][0]
                    value_score += chatround["accept_score"][1]
                    interest_score += chatround["accept_score"][2]
                    count += 1
    
    print('total num: {}  regen num: {}'.format(count, regen_rate))
    # print('average regen num: {}'.format(regen_sum / count))
    print('regen rate: {}'.format(regen_rate / count))
    print('raise rate: {}'.format(raise_rate / regen_rate))
    print('average regen num: {}'.format(regen_sum / count))
    return {"relevance": relevance_score / count, "value": value_score / count, "interest": interest_score / count}

       
def split_list(input_list, total_splits, split_index):
    split_size = len(input_list) // total_splits
    start_index = split_size * (split_index - 1)
    end_index = start_index + split_size if split_index != total_splits else len(input_list)
    return input_list[start_index:end_index]



# 训练采集数据
def train(split = 1, split_index = 1,  result_savepath = ''):
    # 读取用户目录
    with open("/hujinwu/wyf/projects/robotchat/bg_prompt/train_expand.json", "r", encoding="utf-8") as file:  
        bg_prompts = json.load(file)
        # random.shuffle(bg_prompts)
    
    bg_prompts = split_list(bg_prompts, split, split_index)
    
    result_list = []
    if os.path.exists(result_savepath):
        with open(result_savepath, "r", encoding="utf-8") as file:
            result_list = json.load(file)
    
    existing_ids = set([x['expand_id'] for x in result_list])
    
    
    # bg_prompts = bg_prompts[:100]
    
    for user_data in tqdm(bg_prompts):
        # 检查这个用户是否已经生成过了
        if user_data['expand_id'] in existing_ids:
            continue
        
        #### 定义智能体 ####
        user_model_list = [1,2,1,2]    # 用户智能体只选1和2
        chatbot_model_list = [3,4,3,4]  # 机器人智能体只选3和4
        critic = GPTModel(api_key=api_key_gpt,model='gpt-3.5-turbo-0125', temperature=0.0)
        # critic = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat{}".format(user_model_list[split_index - 1]), temperature=0.0)
        # user = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat{}".format(user_model_list[split_index - 1]), temperature=0.75)
        # user = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat1", temperature=0.75)
        chatbot = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat{}".format(user_model_list[split_index - 1]), temperature=0.75)
        # chatbot = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat3", temperature=0.75)
        user = InfiniModel(api_key=api_key_infini_list, model='qwen1.5-72b-chat', temperature=0.75)

        # chatbot = AliyunModel(api_key = api_key_aliyun, model = "qwen1.5-72b-chat", temperature=0.75)
        # user = AliyunModel(api_key = api_key_aliyun, model = "qwen1.5-72b-chat", temperature=0.75)
        # # 以下为使用新部署方式
        # user_model_list = [0,1,0,1]    # 用户智能体只选0和1
        # chatbot_model_list = [2,3,2,3]  # 机器人智能体只选2和3

        # user = OurModel_lf(api_key = "http://localhost:800{}/v1/chat/completions".format(user_model_list[split_index - 1]), model = "string", temperature=0.75)
        
        # chatbot = OurModel_lf(api_key = "http://localhost:800{}/v1/chat/completions".format(chatbot_model_list[split_index - 1]), model = "string", temperature=0.75)

        # critic = GPTModel(api_key=api_key_gpt,model='gpt-3.5-turbo-0125', temperature=0.0)
        
        ####
          
        user.add_system(user_system.format(name = user_data["name"], bg = user_data["prompt"]))
        chatbot.add_system(chatbot_system)
        dialogue_history = []

        record = autochat(chatbot, user, critic, hello_prompt, user_data["name"], user_data["prompt"],
                          dialogue_history, total_chat_round= 5, evaluation= True, regen = True)
        
        if record is not None:
            result_list.append({
                'expand_id': user_data['expand_id'],
                'stage': user_data['stage'],
                'id': user_data['id'],
                'name': record['name'],
                'background': record['background'],
                'conversation': record['conversation']
            })
            
            record_log(result_list, save_path = result_savepath)
    
    # 效果验证
    print('*'*50)
    print(calculate_user_perception_score(result_list))
    print('*'*50)
    print('数据采集数量：{}'.format(len(result_list)))
    
        

def validation(split = 1, split_index = 1,  result_savepath = '', test_only = False):
    # 读取用户目录
    with open("/hujinwu/wyf/projects/robotchat/bg_prompt/validation_expand.json", "r", encoding="utf-8") as file:  
        bg_prompts = json.load(file)
        # random.shuffle(bg_prompts)
    
    bg_prompts = split_list(bg_prompts, split, split_index)
    
    result_list = []
    if os.path.exists(result_savepath):
        with open(result_savepath, "r", encoding="utf-8") as file:
            result_list = json.load(file)
    
    if not test_only:
    
        existing_ids = set([x['expand_id'] for x in result_list])
        
        for user_data in tqdm(bg_prompts):
            # 检查这个用户是否已经生成过了
            if user_data['expand_id'] in existing_ids:
                continue
            
            #### 定义智能体 ####
            user_model_list = [1,2,1,2]    # 用户智能体只选1和2
            chatbot_model_list = [3,4,3,4]  # 机器人智能体只选3和4
            # critic = GPTModel(api_key=api_key_gpt,model='gpt-3.5-turbo-0125', temperature=0.0)
            critic = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat{}".format(user_model_list[split_index - 1]), temperature=0.0)
            user = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat{}".format(user_model_list[split_index - 1]), temperature=0.75)
            # user = InfiniModel(api_key=api_key_infini_list, model='qwen1.5-72b-chat', temperature=0.75)
            chatbot = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat{}".format(chatbot_model_list[split_index - 1]), temperature=0.75)
            ####
            
            user.add_system(user_system.format(name = user_data["name"], bg = user_data["prompt"]))
            chatbot.add_system(chatbot_system)
            dialogue_history = []

            record = autochat(chatbot, user, critic, hello_prompt, user_data["name"], user_data["prompt"],
                            dialogue_history, total_chat_round= 5, evaluation= True, regen = False)
            
            if record is not None:
                result_list.append({
                    'expand_id': user_data['expand_id'],
                    'stage': user_data['stage'],
                    'id': user_data['id'],
                    'name': record['name'],
                    'background': record['background'],
                    'conversation': record['conversation']
                })
                
                record_log(result_list, save_path = result_savepath)
    
    # 效果验证
    print('*'*50)
    print(calculate_user_perception_score(result_list))
    print('*'*50)
    print('数据采集数量：{}'.format(len(result_list)))
    

def test(split = 1, split_index = 1,  result_savepath = '', test_only = False):
    # 读取用户目录
    with open("/hujinwu/wyf/projects/robotchat/bg_prompt/test_expand.json", "r", encoding="utf-8") as file:  
        bg_prompts = json.load(file)
        # random.shuffle(bg_prompts)

    bg_prompts = split_list(bg_prompts, split, split_index)
    # random.shuffle(bg_prompts)
    # bg_prompts = bg_prompts[:200]
    
    result_list = []
    if os.path.exists(result_savepath):
        with open(result_savepath, "r", encoding="utf-8") as file:
            result_list = json.load(file)
    
    if not test_only:
    
        existing_ids = set([x['expand_id'] for x in result_list])
        
        for user_data in tqdm(bg_prompts):
            # 检查这个用户是否已经生成过了
            if user_data['expand_id'] in existing_ids:
                continue
            
            #### 定义智能体 ####
            user_model_list = [1,2,1,2]    # 用户智能体只选1和2
            chatbot_model_list = [3,4,3,4]  # 机器人智能体只选3和4
            critic = GPTModel(api_key=api_key_gpt,model='gpt-3.5-turbo-0125', temperature=0.0)
            # critic = InfiniModel(api_key=api_key_infini_list, model='qwen1.5-72b-chat', temperature=0.75)     
            user = OurModel(api_key = "http://127.0.0.1:7862/chat/chat", model = "Qwen1.5-72B-Chat2", temperature=0.75)
            # user = InfiniModel(api_key=api_key_infini_list, model='qwen1.5-72b-chat', temperature=0.75)
            chatbot = OurModel(api_key = "http://127.0.0.1:7862/chat/chat", model = "Qwen1.5-72B-Chat1", temperature=0.75)
            # chatbot = OurModel_lf(api_key = "http://localhost:8002/v1/chat/completions", model = "string", temperature=0.75, max_tokens=128)
            # chatbot = GPTModel(api_key=api_key_gpt4_eylink, url = 'https://gtapi.xiaoerchaoren.com:8932/v1', model='gpt-4-turbo-preview', temperature=0.75, use_openai=True)
            # chatbot = GPTModel(api_key=api_key_gpt,model='gpt-4o-2024-08-06', temperature=0.75)
            
            # chatbot = NvidiaModel(api_key=api_key_nvidia, url= 'https://integrate.api.nvidia.com/v1/chat/completions', model='meta/llama3-70b-instruct', temperature=0.75)
            
            ####
            
            user.add_system(user_system.format(name = user_data["name"], bg = user_data["prompt"]))
            chatbot.add_system(chatbot_system)
            dialogue_history = []

            record = autochat(chatbot, user, critic, hello_prompt, user_data["name"], user_data["prompt"],
                            dialogue_history, total_chat_round= 5, evaluation= True, regen = False)
            
            if record is not None:
                result_list.append({
                    'expand_id': user_data['expand_id'],
                    'stage': user_data['stage'],
                    'id': user_data['id'],
                    'name': record['name'],
                    'background': record['background'],
                    'conversation': record['conversation']
                })
                
                record_log(result_list, save_path = result_savepath)
    
    # 效果验证
    print('*'*50)
    print(calculate_user_perception_score(result_list))
    # print(calculate_user_perception_score_regen_stat(result_list))
    print('*'*50)
    print('数据采集数量：{}'.format(len(result_list)))
    
    
def test_GLM4(split = 1, split_index = 1,  result_savepath = '', test_only = False):
    # 读取用户目录
    with open("/hujinwu/wyf/projects/robotchat/bg_prompt/test_expand.json", "r", encoding="utf-8") as file:  
        bg_prompts = json.load(file)
        random.shuffle(bg_prompts)

    bg_prompts = split_list(bg_prompts, split, split_index)
    
    result_list = []
    if os.path.exists(result_savepath):
        with open(result_savepath, "r", encoding="utf-8") as file:
            result_list = json.load(file)
    
    if not test_only:
    
        existing_ids = set([x['expand_id'] for x in result_list])
        
        for user_data in tqdm(bg_prompts):
            # 检查这个用户是否已经生成过了
            if user_data['expand_id'] in existing_ids:
                continue
            
            #### 定义智能体 ####
            user_model_list = [1,2,1,2]    # 用户智能体只选1和2
            chatbot_model_list = [3,4,3,4]  # 机器人智能体只选3和4
            critic = GPTModel(api_key=api_key_gpt,model='gpt-3.5-turbo-0125', temperature=0.0)
            user = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat{}".format(user_model_list[split_index - 1]), temperature=0.75)
            # chatbot = GLM4Model(api_key=api_key_glm4,model='glm-4', temperature=0.75)
            chatbot = GPTModel(api_key=api_key_gpt,model='gpt-3.5-turbo-0125', temperature=0.75)
            
            # user = InfiniModel(api_key=api_key_infini_list, model='qwen1.5-72b-chat', temperature=0.75)
            # chatbot = OurModel(api_key = "http://127.0.0.1:7861/chat/chat", model = "Qwen1.5-72B-Chat{}".format(chatbot_model_list[split_index - 1]), temperature=0.75)
            ####
            
            user.add_system(user_system.format(name = user_data["name"], bg = user_data["prompt"]))
            chatbot.add_system(chatbot_system)
            dialogue_history = []

            record = autochat(chatbot, user, critic, hello_prompt, user_data["name"], user_data["prompt"],
                            dialogue_history, total_chat_round= 5, evaluation= True, regen = False)
            
            if record is not None:
                result_list.append({
                    'expand_id': user_data['expand_id'],
                    'stage': user_data['stage'],
                    'id': user_data['id'],
                    'name': record['name'],
                    'background': record['background'],
                    'conversation': record['conversation']
                })
                
                record_log(result_list, save_path = result_savepath)
    
    # 效果验证
    print('*'*50)
    print(calculate_user_perception_score(result_list))
    print('*'*50)
    print('数据采集数量：{}'.format(len(result_list)))
    

if __name__ == '__main__':
    # os.system("clear")
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--stage", type=str, default='test') # 数据集收集阶段，可选项：train validation test
    parser.add_argument("--split", type=int, default= 1) # 对数据集合进行分割，每个集合分别处理
    parser.add_argument("--split_index", type=int, default= 1)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--result_path", type=str, default='/hujinwu/wyf/projects/robotchat/results/gpt4o/gpt4o_test_new.json')
    # 可以加一个：每个split用户智能体和机器人智能体的选择调用模型列表
    
    
    # 明天使用的chatbot模型：
    '''
    /hujinwu/LLM_Assemble/pretrain_model/models--Qwen--Qwen1.5-32B-Chat/snapshots/0b1785d88bbe93aa90a8a19da8af78eccbf010a6   第一轮
    /hujinwu/code/robotchat/Firefly/script/checkpoint/firefly-qwen1.5-32b-sft-qlora_cl_s01
    /hujinwu/code/robotchat/Firefly/script/checkpoint/firefly-qwen1.5-32b-sft-qlora_cl_s02
    /hujinwu/code/robotchat/Firefly/script/checkpoint/firefly-qwen1.5-32b-sft-qlora_cl_s03
    '''
    
    
    args = parser.parse_args()
    
    
    if args.stage == 'train':
        train(split=args.split, split_index=args.split_index, result_savepath = args.result_path)

    elif args.stage == 'validation':
        validation(split=args.split, split_index=args.split_index, result_savepath = args.result_path, test_only=args.test_only)

    elif args.stage == 'test':
        test(split=args.split, split_index=args.split_index, result_savepath = args.result_path, test_only=args.test_only)
        # test_GLM4(split=args.split, split_index=args.split_index, result_savepath = args.result_path, test_only=args.test_only)



# 测试结果，只需要模拟一次测试时采集数据就行
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage test  --split 1 --split_index 1 --test_only --result_path /hujinwu/wyf/projects/robotchat/results/vicuna33b_cl/vicuna33b_cl_test_s04.json 就用这个数据
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage validation  --split 1 --split_index 1 --test_only --result_path /hujinwu/wyf/projects/robotchat/results/qwen72b_cl/qwen_cl_val_s03.json


# 训练采数据   需要确认的内容：部署的模型，保存路径
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage train  --split 4 --split_index 1 --result_path /hujinwu/wyf/projects/robotchat/results/qwen32b_cl/qwen_cl_train_s03_1.json
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage train  --split 4 --split_index 2 --result_path /hujinwu/wyf/projects/robotchat/results/qwen32b_cl/qwen_cl_train_s03_2.json
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage train  --split 4 --split_index 3 --result_path /hujinwu/wyf/projects/robotchat/results/qwen32b_cl/qwen_cl_train_s03_3.json
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage train  --split 4 --split_index 4 --result_path /hujinwu/wyf/projects/robotchat/results/qwen32b_cl/qwen_cl_train_s03_4.json

# 验证采数据   需要确认的内容：部署的模型，保存路径
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage validation  --split 4 --split_index 1  --result_path /hujinwu/wyf/projects/robotchat/results/qwen32b_cl/qwen_cl_val_s02_1.json
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage validation  --split 4 --split_index 2  --result_path /hujinwu/wyf/projects/robotchat/results/qwen32b_cl/qwen_cl_val_s02_2.json
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage validation  --split 4 --split_index 3  --result_path /hujinwu/wyf/projects/robotchat/results/qwen32b_cl/qwen_cl_val_s02_3.json
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage validation  --split 4 --split_index 4  --result_path /hujinwu/wyf/projects/robotchat/results/qwen32b_cl/qwen_cl_val_s02_4.json

# 测试采数据   需要确认的内容：部署的模型，保存路径
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage test  --split 4 --split_index 1 --result_path /hujinwu/wyf/projects/robotchat/results/llama3_70b_cl/llama3_70b_cl_test_s00_1.json
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage test  --split 4 --split_index 2 --result_path /hujinwu/wyf/projects/robotchat/results/llama3_70b_cl/llama3_70b_cl_test_s00_2.json
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage test  --split 4 --split_index 3 --result_path /hujinwu/wyf/projects/robotchat/results/llama3_70b_cl/llama3_70b_cl_test_s00_3.json
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage test  --split 4 --split_index 4 --result_path /hujinwu/wyf/projects/robotchat/results/llama3_70b_cl/llama3_70b_cl_test_s00_4.json




# 测试GLM-4 GPT-3.5 Turbo
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage test  --split 4 --split_index 1 --result_path /hujinwu/wyf/projects/robotchat/results/gpt3.5turbo/gpt3.5_test_1.json 
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage test  --split 4 --split_index 2 --result_path /hujinwu/wyf/projects/robotchat/results/gpt3.5turbo/gpt3.5_test_2.json 
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage test  --split 4 --split_index 3 --result_path /hujinwu/wyf/projects/robotchat/results/gpt3.5turbo/gpt3.5_test_3.json 
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage test  --split 4 --split_index 4 --result_path /hujinwu/wyf/projects/robotchat/results/gpt3.5turbo/gpt3.5_test_4.json 



    
    

# 某一轮：采集、训练、测试
# python /hujinwu/wyf/projects/robotchat/data_gen.py --stage train  --split 4 --split_index 1 --result_path /hujinwu/wyf/projects/robotchat/results/qwen32b_cl/qwen_cl_train_s03_1.json