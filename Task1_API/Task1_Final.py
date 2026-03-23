import fitz  # PyMuPDF
from openai import OpenAI
import json
import re

# 1. 配置部分 (使用你截图中的 Key)
API_KEY = "sk-2a897650ef2c4fb79d37af2e51cc43f4"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 2. 路径部分 (请确保文件名完全正确)
pdf_path = r"C:\Users\LENOVO\Desktop\A case of portal vein recanalization and symptomatic heart failure.pdf"

def get_pdf_text(path):
    """从 PDF 提取文本"""
    doc = fitz.open(path)
    return "".join([page.get_text() for page in doc])

def extract_entities(text):
    """通过大模型提取医疗实体"""
    prompt = f"请从以下中文病例中提取信息，并以 JSON 格式返回。包含：患者基本信息、主要症状、既往史、诊断结果、治疗方案。只返回 JSON 代码块：\n\n{text}"
    
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- 执行流程 ---
try:
    print("1. 正在读取 PDF...")
    content = get_pdf_text(pdf_path)
    
    print("2. 正在调用 API (通义千问)...")
    raw_json = extract_entities(content)
    
    # 清洗并解析 JSON
    clean_json = re.sub(r'```json|```', '', raw_json).strip()
    result_data = json.loads(clean_json)
    
    print("3. 提取成功！结果如下：")
    print(json.dumps(result_data, ensure_ascii=False, indent=4))
    
    # 保存文件
    with open("extracted_case.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=4)
    print("\n✅ 文件已保存为 extracted_case.json")

except Exception as e:
    print(f"\n❌ 出错了: {e}")
