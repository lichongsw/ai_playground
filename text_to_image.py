import requests
import json
import os
import sys
import time
import base64
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# 设置代理（如需要）
os.environ.update({'http_proxy': 'http://127.0.0.1:7890', 'https_proxy': 'http://127.0.0.1:7890'})

# 请在此处填入您的 Gemini API 密钥
API_KEY = "AIzaSyBIbskgZ5_35l5p5JzMLWd8lh-NykPlVbs"  # 替换为您的实际 Gemini API 密钥

def get_mime_type(file_path):
    """根据文件扩展名获取MIME类型"""
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/jpeg')

def image_to_base64(image_path):
    """将图片文件转换为base64编码"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

def select_image_files():
    """打开文件选择对话框并返回选择的多个图片路径"""
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.attributes('-topmost', True)  # 确保对话框在最前面
    
    file_paths = filedialog.askopenfilenames(
        title="选择参考图片（可多选）",
        filetypes=[
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
            ("所有文件", "*.*")
        ]
    )
    
    root.destroy()
    return file_paths

def save_base64_image(base64_data, output_path):
    """将 base64 编码的图片保存到文件"""
    # 从数据 URL 中提取 base64 部分
    if "base64," in base64_data:
        base64_data = base64_data.split("base64,")[1]
    
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(base64_data))
    
    return output_path

def ask_text_to_image_mode():
    """询问用户是否使用文生图模式"""
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno("文生图模式", "未选择图片。是否使用文生图模式？")
    root.destroy()
    return result

def get_text_prompt(default_prompt="生成一个可爱的小猫咪", is_text_to_image=False):
    """获取用户输入的文本提示词，优化UI样式"""
    root = tk.Tk()
    root.title("输入提示词" if not is_text_to_image else "输入文生图提示词")
    
    # 设置窗口大小和位置
    window_width = 500
    window_height = 180
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # 创建框架来包含所有元素
    main_frame = tk.Frame(root, padx=10, pady=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # 输入框，减少内边距
    text_entry = tk.Text(main_frame, height=5, width=50, padx=5, pady=5)
    text_entry.pack(fill=tk.BOTH, expand=True)
    text_entry.insert("1.0", default_prompt)
    
    # 创建底部按钮框架
    button_frame = tk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=(5, 0))
    
    prompt_result = {"text": ""}
    
    def on_submit():
        prompt_result["text"] = text_entry.get("1.0", "end-1c")
        root.destroy()
    
    # 按钮放在右侧
    submit_button = tk.Button(button_frame, text="提交", command=on_submit, width=10)
    submit_button.pack(side=tk.RIGHT)
    
    # 聚焦到文本框并设置光标位置
    text_entry.focus_set()
    text_entry.mark_set("insert", "end")
    
    # 绑定回车键提交（按Ctrl+Enter提交）
    def on_ctrl_enter(event):
        on_submit()
        return "break"  # 阻止默认行为
    
    text_entry.bind("<Control-Return>", on_ctrl_enter)
    
    root.mainloop()
    return prompt_result["text"]

def main():
    # 弹出文件选择对话框
    print("请选择一个或多个参考图片文件...")
    image_paths = select_image_files()
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建输出目录
    output_dir = os.path.join(script_dir, "Gemini Images")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = int(time.time())
    
    # 检查是否选择了图片
    if not image_paths:
        # 询问是否使用文生图模式
        use_text_to_image = ask_text_to_image_mode()
        
        if not use_text_to_image:
            print("未选择图片且不使用文生图模式，退出程序。")
            return
        
        # 获取用户输入的提示词
        prompt_text = get_text_prompt("", True)
        if not prompt_text.strip():
            print("未输入提示词，退出程序。")
            return
        
        print(f"使用文生图模式，提示词: {prompt_text}")
        
        # 构建文生图请求数据
        request_data = {
            "contents": [{
                "parts":[
                    {"text": prompt_text}
                ]
            }],
            "generationConfig": {"responseModalities": ["Text", "Image"]}
        }
        
        # 生成文件名
        request_id = f"text_to_image_{timestamp}"
    else:
        print(f"已选择 {len(image_paths)} 张图片")
        
        # 获取第一个文件的名称作为请求标识
        first_file_name = os.path.splitext(os.path.basename(image_paths[0]))[0]
        if len(image_paths) > 1:
            request_id = f"{first_file_name}_and_{len(image_paths)-1}_more_{timestamp}"
        else:
            request_id = f"{first_file_name}_{timestamp}"
        
        # 获取用户输入的提示词
        default_prompt = ""
        prompt_text = get_text_prompt(default_prompt)
        
        # 构建多图像请求数据
        parts = [{"text": prompt_text}]
        
        # 添加所有图像到请求
        for image_path in image_paths:
            try:
                base64_image = image_to_base64(image_path)
                mime_type = get_mime_type(image_path)
                
                # 添加图像部分
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_image
                    }
                })
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {e}")
        
        # 构建完整请求
        request_data = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {"responseModalities": ["Text", "Image"]}
        }
    
    # 发送请求
    print("正在发送请求到 Gemini API...")
    try:
        api_url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key={API_KEY}'
        
        response = requests.post(
            api_url,
            headers={
                'Content-Type': 'application/json'
            },
            json=request_data,
            proxies={
                'http': 'http://127.0.0.1:7890',
                'https': 'http://127.0.0.1:7890'
            }
        )
        
        # 保存请求数据用于调试
        debug_request_file = os.path.join(output_dir, f"{request_id}_request.json")
        with open(debug_request_file, 'w', encoding='utf-8') as f:
            # 移除base64数据以避免文件过大，仅用于调试
            debug_data = request_data.copy()
            if image_paths:  # 只有在有图片时才需要移除base64数据
                for i in range(1, len(debug_data["contents"][0]["parts"])):
                    if "inline_data" in debug_data["contents"][0]["parts"][i]:
                        debug_data["contents"][0]["parts"][i]["inline_data"]["data"] = "[BASE64_DATA_REMOVED_FOR_DEBUGGING]"
            json.dump(debug_data, f, ensure_ascii=False, indent=2)
        
        # 检查响应状态
        if response.status_code != 200:
            print(f"API返回错误: {response.status_code}")
            print(f"错误详情: {response.text}")
            
            # 保存错误响应
            error_file = os.path.join(output_dir, f"{request_id}_error.json")
            with open(error_file, 'w', encoding='utf-8') as f:
                try:
                    error_json = response.json()
                    json.dump(error_json, f, ensure_ascii=False, indent=2)
                except:
                    f.write(response.text)
            
            print(f"错误信息已保存到: {error_file}")
            sys.exit(1)
        
        # 解析响应
        result = response.json()
        
        # 保存完整响应到 JSON 文件
        response_file = os.path.join(output_dir, f"{request_id}_response.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 尝试提取和保存图片
        try:
            # 根据Gemini API响应结构，提取图像数据
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    
                    # 遍历parts寻找图片数据
                    image_count = 0
                    for i, part in enumerate(parts):
                        # 检查是否有inlineData字段（注意大小写与响应一致）
                        if "inlineData" in part and "data" in part["inlineData"]:
                            image_count += 1
                            image_data = part["inlineData"]["data"]
                            image_type = part["inlineData"].get("mimeType", "image/png")
                            ext = image_type.split("/")[-1]
                            
                            # 保存图片，使用与请求/响应文件相同的命名格式，为多图片添加序号
                            image_file = os.path.join(output_dir, f"{request_id}_generated_{image_count}.{ext}")
                            save_base64_image(image_data, image_file)
                            print(f"生成成功！图片 {image_count} 已保存到: {image_file}")
                    
                    if image_count == 0:
                        print("响应中未找到图片数据，请检查响应JSON文件")
                    else:
                        print(f"共保存了 {image_count} 张生成的图片")
                else:
                    print("响应结构中未找到content或parts字段，请检查响应JSON文件")
            else:
                print("响应中未找到candidates字段，请检查响应JSON文件")
            
            # 打开保存目录
            print(f"所有文件已存储到: {output_dir}")
            os.startfile(output_dir)
            
        except Exception as e:
            print(f"处理响应时出错: {e}")
            print("请查看保存的 JSON 文件以了解完整响应结构")
        
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()