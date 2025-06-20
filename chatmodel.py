import time 
import requests
import json
from openai import OpenAI
import base64
from openai import OpenAI
from io import BytesIO
from PIL import Image
import os

global_call_num = 0




class AliyunModel():
    def __init__(self, model = 'Qwen/Qwen2.5-VL-72B-Instruct', apikey = None) -> None:
        # self.api_key = os.environ.get('apikey')
        self.api_key = apikey
        self.model = model
        self.base_url = 'https://api-inference.modelscope.cn/v1'
        self.messages = []  # 这个暂时用不上
 
    def call(self, content, image_PIL=None, temperature=0.0, max_tokens = 2048, top_p=0.95, frequency_penalty=0, presence_penalty=0, add_messages = False):   # 这里content是一个列表，而不是单次对话

        apikey = None
        if isinstance(self.api_key, str):
            apikey = self.api_key
        elif isinstance(self.api_key, list):
            apikey = self.api_key[global_call_num % len(self.api_key)]
            
        else:
            raise ValueError

        client = OpenAI(
            api_key = apikey, # 请替换成您的ModelScope SDK Token
            base_url = self.base_url
        )
        # 将图片转为base64格式
        if image_PIL is not None:
            # 如果是单个图片
            if isinstance(image_PIL, Image.Image):
                buffered = BytesIO()
                image_PIL.save(buffered, format="PNG")
                image_url = ["data:image/png;base64,"+ base64.b64encode(buffered.getvalue()).decode('utf-8')]
            elif isinstance(image_PIL, list):
                image_url = []
                for image in image_PIL:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    image_url.append("data:image/png;base64,"+ base64.b64encode(buffered.getvalue()).decode('utf-8'))
            else:
                raise ValueError('image_PIL should be a PIL image or a list of PIL images')
            
        
            # 如果是单个图片
            if isinstance(image_PIL, Image.Image):
                messages=[{
                    "role": "user",
                    "content": [
                        # NOTE: 使用图像令牌 <image> 的提示格式是不必要的，因为提示将由API服务器自动处理。
                        # 由于提示将由API服务器自动处理，因此不需要使用包含 <image> 图像令牌的提示格式。
                        {"type": "text", "text": content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url[0],
                            },
                        },
                    ],
                }]
            elif isinstance(image_PIL, list):
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content}
                    ],
                }]
                for i in range(len(image_PIL)):
                    messages[0]['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url[i],
                        },
                    })
        else:
            messages = [
            # {
            #     "role": "system",
            #     "content": [
            #         {"type": "text", "text": "You are a helpful assistant."}   # 注意这里的system和前面有图片的不一样，这里更像一个LLM的提示
            #     ],
            # },
            {
                "role": "user",
                "content": [
                    {   "type": "text", 
                        "text": content
                    },
                ],
            }
            ]            


        # 最多尝试50次，否则报错
        post_success = False
        for try_num in range(5):
        
            if try_num >= 1:
                print('尝试第{}次调用api，'.format(try_num + 1))
            try: 
                response = client.chat.completions.create(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, top_p =top_p,  frequency_penalty = frequency_penalty, presence_penalty = presence_penalty, stream=False)
                post_success = True
                break
                
            except:
                # global_call_num += 1
                # apikey = self.api_key[global_call_num % len(self.api_key)]
                time.sleep(2*(try_num+1))  # 是否需要延时
            
        if post_success:
            return response.choices[0].message.content
        else:
            print('多次调用api，但无法返回结果')
            raise ValueError
                 
    def add_system(self, system):
        self.messages = [{'role': 'system', 'content': system}]
    
    def add_content(self, content):
        self.messages.append({'role': 'assistant', 'content': content})

    
    def clear(self):
        self.messages = []


class GPTModel():
    def __init__(self, model = 'gpt-4-1106-preview', apikey = None) -> None:
        # self.api_key = os.environ.get('apikey')
        self.api_key = apikey
        self.model = model
        self.base_url = 'https://api.openai-proxy.org/v1'
        self.messages = []  # 这个暂时用不上
 
    def call(self, content, image_PIL=None, temperature=0.2, max_tokens = 32, add_messages = False):   # 这里content是一个列表，而不是单次对话

        apikey = None
        if isinstance(self.api_key, str):
            apikey = self.api_key
        elif isinstance(self.api_key, list):
            apikey = self.api_key[global_call_num % len(self.api_key)]
            
        else:
            raise ValueError

        client = OpenAI(
            api_key = apikey, # 请替换成您的ModelScope SDK Token
            base_url = self.base_url
        )
        messages = [
        {
            "role": "user",
            "content": content
        }
        ]            

        # 最多尝试5次，否则报错
        post_success = False
        for try_num in range(5):
        
            if try_num >= 1:
                print('尝试第{}次调用api，'.format(try_num + 1))
            try: 
                response = client.chat.completions.create(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, seed = 1234, stream=False)
                post_success = True
                break
                
            except:
                # global_call_num += 1
                # apikey = self.api_key[global_call_num % len(self.api_key)]
                time.sleep(2*(try_num+1))  # 是否需要延时
            
        if post_success:
            return response.choices[0].message.content
        else:
            print('多次调用api，但无法返回结果')
            raise ValueError
                 
    def add_system(self, system):
        self.messages = [{'role': 'system', 'content': system}]
    
    def add_content(self, content):
        self.messages.append({'role': 'assistant', 'content': content})
    
    def clear(self):
        self.messages = []



class Internlm_Model():
    def __init__(self, model = 'internvl2.5-78b') -> None:
        self.api_key = os.environ.get('apikey_InternLM')
        self.model = model
        self.base_url = 'https://chat.intern-ai.org.cn/api/v1/'
        self.messages = []  # 这个暂时用不上
 
    def call(self, content, image_PIL=None, temperature=0.0, max_tokens = 2048, top_p=0.95, frequency_penalty=0, presence_penalty=0, add_messages = False):   # 这里content是一个列表，而不是单次对话

        apikey = None
        if isinstance(self.api_key, str):
            apikey = self.api_key
        elif isinstance(self.api_key, list):
            apikey = self.api_key[global_call_num % len(self.api_key)]
            
        else:
            raise ValueError

        client = OpenAI(
            api_key = apikey, # 请替换成您的ModelScope SDK Token
            base_url = self.base_url
        )
        # 将图片转为base64格式
        if image_PIL is not None:
            # 如果是单个图片
            if isinstance(image_PIL, Image.Image):
                buffered = BytesIO()
                image_PIL.save(buffered, format="PNG")
                image_url = ["data:image/png;base64,"+ base64.b64encode(buffered.getvalue()).decode('utf-8')]
            elif isinstance(image_PIL, list):
                image_url = []
                for image in image_PIL:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    image_url.append("data:image/png;base64,"+ base64.b64encode(buffered.getvalue()).decode('utf-8'))
            else:
                raise ValueError('image_PIL should be a PIL image or a list of PIL images')
            
        
            # 如果是单个图片
            if isinstance(image_PIL, Image.Image):
                messages=[{
                    "role": "user",
                    "content": [
                        # NOTE: 使用图像令牌 <image> 的提示格式是不必要的，因为提示将由API服务器自动处理。
                        # 由于提示将由API服务器自动处理，因此不需要使用包含 <image> 图像令牌的提示格式。
                        {"type": "text", "text": content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url[0],
                            },
                        },
                    ],
                }]
            elif isinstance(image_PIL, list):
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content}
                    ],
                }]
                for i in range(len(image_PIL)):
                    messages[0]['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url[i],
                        },
                    })
        else:
            messages = [
            # {
            #     "role": "system",
            #     "content": [
            #         {"type": "text", "text": "You are a helpful assistant."}   # 注意这里的system和前面有图片的不一样，这里更像一个LLM的提示
            #     ],
            # },
            {
                "role": "user",
                "content": [
                    {   "type": "text", 
                        "text": content
                    },
                ],
            }
            ]            


        # 最多尝试50次，否则报错
        post_success = False
        for try_num in range(5):
        
            if try_num >= 1:
                print('尝试第{}次调用api，'.format(try_num + 1))
            try: 
                response = client.chat.completions.create(model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens, top_p =top_p,  frequency_penalty = frequency_penalty, presence_penalty = presence_penalty, stream=False)
                post_success = True
                break
                
            except:
                # global_call_num += 1
                # apikey = self.api_key[global_call_num % len(self.api_key)]
                time.sleep(2*(try_num+1))  # 是否需要延时
            
        if post_success:
            return response.choices[0].message.content
        else:
            print('多次调用api，但无法返回结果')
            raise ValueError
                 
    def add_system(self, system):
        self.messages = [{'role': 'system', 'content': system}]
    
    def add_content(self, content):
        self.messages.append({'role': 'assistant', 'content': content})

    
    def clear(self):
        self.messages = []





class InfiniModel():
    def __init__(self, model) -> None:
        self.api_key = os.environ.get('apikey')
        self.api_key = "sk-c7a4yzsw7iwhlvbk"
        self.model = model
        self.base_url = 'https://cloud.infini-ai.com/maas/v1'
 
    def call(self, content, image_PIL=None, temperature=0.0, max_tokens = 2048, top_p=0.95, frequency_penalty=0, presence_penalty=0, add_messages = False):   # 这里content是一个列表，而不是单次对话
        global global_call_num

        apikey = None
        if isinstance(self.api_key, str):
            apikey = self.api_key
        elif isinstance(self.api_key, list):
            apikey = self.api_key[global_call_num % len(self.api_key)]
            
        else:
            raise ValueError

        client = OpenAI(
            api_key = apikey, # 请替换成您的ModelScope SDK Token
            base_url = self.base_url
        )
        # 将图片转为base64格式
        if image_PIL is not None:
            # 如果是单个图片
            if isinstance(image_PIL, Image.Image):
                buffered = BytesIO()
                image_PIL.save(buffered, format="PNG")
                image_url = ["data:image/png;base64,"+ base64.b64encode(buffered.getvalue()).decode('utf-8')]
            elif isinstance(image_PIL, list):
                image_url = []
                for image in image_PIL:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    image_url.append("data:image/png;base64,"+ base64.b64encode(buffered.getvalue()).decode('utf-8'))
            else:
                raise ValueError('image_PIL should be a PIL image or a list of PIL images')
            
        
            # 如果是单个图片
            if isinstance(image_PIL, Image.Image):
                messages=[{
                    "role": "user",
                    "content": [
                        # NOTE: 使用图像令牌 <image> 的提示格式是不必要的，因为提示将由API服务器自动处理。
                        # 由于提示将由API服务器自动处理，因此不需要使用包含 <image> 图像令牌的提示格式。
                        {"type": "text", "text": content},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url[0],
                            },
                        },
                    ],
                }]
            elif isinstance(image_PIL, list):
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content}
                    ],
                }]
                for i in range(len(image_PIL)):
                    messages[0]['content'].append({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url[i],
                        },
                    })
        else:
            messages = [
            # {
            #     "role": "system",
            #     "content": [
            #         {"type": "text", "text": "You are a helpful assistant."}   # 注意这里的system和前面有图片的不一样，这里更像一个LLM的提示
            #     ],
            # },
            {
                "role": "user",
                "content": [
                    {   "type": "text", 
                        "text": content
                    },
                ],
            }
            ]            


        # 最多尝试50次，否则报错
        post_success = False
        for try_num in range(5):
        
            if try_num >= 1:
                print('尝试第{}次调用api，'.format(try_num + 1))
            try: 
                response = client.chat.completions.create(model=self.model.lower(), messages=messages, temperature=temperature, max_tokens=max_tokens,  top_p =top_p,  frequency_penalty = frequency_penalty, presence_penalty = presence_penalty, stream=False)
                post_success = True
                break
                
            except:
                # global_call_num += 1
                # apikey = self.api_key[global_call_num % len(self.api_key)]
                time.sleep(2*(try_num+1))  # 是否需要延时
            
        if post_success:
            return response.choices[0].message.content
        else:
            print('多次调用api，但无法返回结果')
            raise ValueError
                 
    def add_system(self, system):
        self.messages = [{'role': 'system', 'content': system}]
    
    def add_content(self, content):
        self.messages.append({'role': 'assistant', 'content': content})

    
    def clear(self):
        self.messages = []

        


class Minicpm_Model():
    def __init__(self,  model, api_key = "token-abc123", url = 'http://localhost:8006/v1') -> None:
        self.api_key = api_key   # 可以为list，也可以为单个字符串
        self.model = model
        self.messages = []
        self.url = url
        self.client = OpenAI(api_key=self.api_key, base_url=self.url)
    
    def call(self, content, image_PIL=None, temperature=0.0,  add_messages = False):

        # 将图片转为base64格式
        if image_PIL is not None:
            # 如果是单个图片
            if isinstance(image_PIL, Image.Image):
                buffered = BytesIO()
                image_PIL.save(buffered, format="PNG")
                image_url = ["data:image/png;base64,"+ base64.b64encode(buffered.getvalue()).decode('utf-8')]
            elif isinstance(image_PIL, list):
                image_url = []
                for image in image_PIL:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    image_url.append("data:image/png;base64,"+ base64.b64encode(buffered.getvalue()).decode('utf-8'))
            else:
                raise ValueError('image_PIL should be a PIL image or a list of PIL images')
        
        
        # 最多尝试50次，否则报错
        post_success = False
        for try_num in range(10):
        
            if try_num >= 1:
                print('retry number: {}'.format(try_num + 1))
            
            # try: 
                # 处理messages
            if image_PIL is not None:
                # 如果是单个图片
                if isinstance(image_PIL, Image.Image):
                    messages=[{
                        "role": "user",
                        "content": [
                            # NOTE: 使用图像令牌 <image> 的提示格式是不必要的，因为提示将由API服务器自动处理。
                            # 由于提示将由API服务器自动处理，因此不需要使用包含 <image> 图像令牌的提示格式。
                            {"type": "text", "text": content},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url[0],
                                },
                            },
                        ],
                    }]
                elif isinstance(image_PIL, list):
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": content}
                        ],
                    }]
                    for i in range(len(image_PIL)):
                        messages[0]['content'].append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_url[i],
                            },
                        })
                    
            else:
                messages=[{
                        "role": "user",
                        "content": [
                            # NOTE: 使用图像令牌 <image> 的提示格式是不必要的，因为提示将由API服务器自动处理。
                            # 由于提示将由API服务器自动处理，因此不需要使用包含 <image> 图像令牌的提示格式。
                            {"type": "text", "text": content}
                        ],
                    }]                    
            # 下面这个函数会一直log输出HTTP Request: POST http://localhost:8006/v1/chat/completions "HTTP/1.1 200 OK"
            chat_response = self.client.chat.completions.create(
                model=self.model, # model_local_path or huggingface id
                temperature=temperature,
                messages=messages,
                extra_body={
                    "stop_token_ids": [151645, 151643]
                }
            )
            
            result = chat_response.choices[0].message.content
            
            if add_messages:
                self.messages.append({'role': 'user', 'content': content})   
                self.messages.append({'role': 'assistant', 'content': result})
                
            post_success = True
            break
            # except:
            #     time.sleep(2)  # 是否需要延时
            
        if post_success:
            return result
        else:
            print('多次调用api，但无法返回结果')
            raise ValueError
                 
    def add_system(self, system):
        self.messages = [{'role': 'system', 'content': system}]
    
    def add_content(self, content):
        self.messages.append({'role': 'assistant', 'content': content})
  
    def clear(self):
        self.messages = []


# # 使用自己构造的web api
# class Minicpm_Model_ours():
#     def __init__(self, url = 'http://127.0.0.1:8000/predict/') -> None:
#         self.url = url

    
#     def call(self, content, image_PIL=None, temperature=0.0, add_messages = False):

#         # 将图片转为base64格式
#         if image_PIL is not None:
#             # 如果是单个图片
#             if isinstance(image_PIL, Image.Image):
#                 buffered = BytesIO()
#                 image_PIL.save(buffered, format="PNG")
#                 image_url = [base64.b64encode(buffered.getvalue()).decode('utf-8')]
#             elif isinstance(image_PIL, list):
#                 image_url = []
#                 for image in image_PIL:
#                     buffered = BytesIO()
#                     image.save(buffered, format="PNG")
#                     image_url.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
#             else:
#                 raise ValueError('image_PIL should be a PIL image or a list of PIL images')
#         else:
#             image_url = []
        
        
#         # 最多尝试50次，否则报错
#         post_success = False
#         # for try_num in range(10):
        
#             # if try_num >= 1:
#             #     print('retry number: {}'.format(try_num + 1))
            
#             # try: 
#                 # 处理messages

#         messages=[{
#             "role": "user",
#             "content": content,
#         }]
#         data = {
#             "temperature": temperature,
#             "msgs": messages,
#             "img_urls": image_url,
#             "max_new_tokens": 2048,
#         }
        
#         headers = {"Content-Type": "application/json"}
#         chat_response = requests.post(self.url, headers=headers, json=data)    # 发送请求           
        
#         result = chat_response.json()['answer']
#         # print(result)
        
#         if add_messages:
#             self.messages.append({'role': 'user', 'content': content})   
#             self.messages.append({'role': 'assistant', 'content': result})
            
#         post_success = True
#                 # break
#         #     except:
#         #         time.sleep(2)  # 是否需要延时
            
#         # if post_success:
#         return result
#         # else:
#         #     print('多次调用api，但无法返回结果')
#         #     raise ValueError
                 
#     def add_system(self, system):
#         self.messages = [{'role': 'system', 'content': system}]
    
#     def add_content(self, content):
#         self.messages.append({'role': 'assistant', 'content': content})
  
#     def clear(self):
#         self.messages = []



if __name__ == '__main__':
    
    # conda activate vllm && vllm serve /share/home/wangyufeng/projects/pretrain_model/Qwen2.5-VL-72B-Instruct-AWQ --served-model-name Qwen2.5-VL-72B-Instruct-AWQ --dtype auto --max-model-len 2048 --api-key token-abc123 --trust-remote-code --port 8006  --gpu_memory_utilization 0.9

    # # 无问芯穹
    # model = InfiniModel(api_key=api_key_infini_list, model='qwen2.5-72b-instruct')
    # answer = model.call([{"role": "user", "content": 'Hi there! I\'m Elena Rodriguez'}])
    # print(answer)
    
    # 本地minicpm
    # CUDA_VISIBLE_DEVICES=1 vllm serve /lichenghao/wyf/model_cache/MiniCPM-V-2_6 --dtype auto --max-model-len 2048 --api-key token-abc123 --gpu_memory_utilization 0.95 --trust-remote-code

    # MLLM_model = Minicpm_Model(model = 'Qwen2.5-VL-72B-Instruct', url = 'http://localhost:8006/v1')
    # MLLM_model = Minicpm_Model(model = '/lichenghao/wyf/model_cache/MiniCPM-V-2_6')
    # MLLM_model = Minicpm_Model(model = '/share/home/wangyufeng/model_cache/MiniCPM-V-2_6')
    # MLLM_model = Minicpm_Model(model = 'MiniCPM-V-2_6', url = 'http://localhost:8006/v1')
    
    # MLLM_model = InfiniModel(model = 'qwen2.5-vl-72b-instruct')

    MLLM_model = AliyunModel(model = 'Qwen/Qwen2.5-VL-72B-Instruct', apikey='f1d4df5a-773d-49ed-9c89-5410de01c7eb')

    # image_path = '/home/wang/habitat-sim/image.png'
    # image = Image.open('/lichenghao/wyf/projects/activeEQA/results/vlm_exp/0/0.png').convert('RGB')
    
    # result = MLLM_model.call(content = "这张图片里有什么", image_PIL=image, temperature=0.0, add_messages = False)
    # print(result)
    

    # MLLM_model = Minicpm_Model_ours()
    # image_list = []
    # # image_list.append(Image.open('/home/wang/habitat-sim/forward.png').convert('RGB'))
    # # image_list.append(Image.open('/home/wang/habitat-sim/right.png').convert('RGB'))
    # result = MLLM_model.call(content = "第一张图和第二张图中分别有什么", image_PIL=None, temperature=0.0, add_messages = False)
    # print(result)
    

    # image_path_list = ['results/vlm_exp_debug/0/0_view0_focus_rgb.png', 
    #                    'results/vlm_exp_debug/0/0_view45_focus_rgb.png',
    #                    'results/vlm_exp_debug/0/0_view315_focus_rgb.png']
    # image_list = [img.resize((640, 640)) for img in [Image.open(path).convert('RGB') for path in image_path_list]]


    image = Image.open('results/vlm_exp_debug/0/9.png').convert('RGB')    
    result = MLLM_model.call(content = '你是谁', image_PIL=image, temperature=0.2)
    print(result)   




