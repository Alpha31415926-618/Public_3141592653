import asyncio
import json
import time
import os
import glob
from datetime import datetime
from loguru import logger
from openai import AsyncOpenAI, APIError, APITimeoutError

# 配置日志
logger.add("openai_llm.log")

NUM = os.getenv("NUM", "1")

LLM_API_URL = os.getenv("LLM_API_URL", "http://0.0.0.0:11434/v1")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
TASK = os.getenv("TASK", "ner&re")

MODEL_NAME = os.getenv("MODEL_NAME", "qwen3")

# 运行方式示例: CONCURRENCY_LIMIT=10 python script.py
try:
    CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "5"))
except ValueError:
    logger.warning("Invalid CONCURRENCY_LIMIT env var, defaulting to 5")
    CONCURRENCY_LIMIT = 100

INPUT_DIR = "./data"
OUTPUT_DIR = "./output"

logger.info(f"Task Config -> URL: {LLM_API_URL}, Model: {MODEL_NAME}, Concurrency: {CONCURRENCY_LIMIT}")
logger.info(f"Directories -> Input: {INPUT_DIR}, Output: {OUTPUT_DIR}")


SYSTEM_PROMPT_TEMPLATE_NER = """
# Role
你是一名资深的严格的生物医药知识图谱构建专家，专注于仿制药开发、临床评价及转化医学领域。你精通从非结构化文本中进行命名实体识别（NER）。
，禁止根据常识推理文中未提及的信息。禁止修正文中的错别字。禁止输出文中不存在的词汇。

# Task
读取用户提供的【待分析文本】，执行以下操作：
在做实体抽取之前，请先分析句子结构，
1.  **实体抽取**：识别文本中具有高价值的生物医药实体，必须是名词或者名词短语
3.  **格式输出**：输出严格的 JSON 格式数据。

# 实体抽取标准：
请在抽取时严格参考以下分类体系：
1.  **药物与CMC (Drug_CMC)**
    *   物质：通用名、商品名、化学名、CAS号、API、辅料、杂质、代谢产物等。
    *   属性：晶型、盐型、溶解度、BCS分类等。
    *   工艺：剂型、给药途径、规格等。
2.  **法规与评价 (Regulatory_Evaluation)**
    *   核心：参比制剂(RLD)、生物等效性(BE)、一致性评价、治疗等效性(TE)、专利挑战(P4)等。
    *   法规：ANDA、DMF、橙皮书、CDE、MAH等。
3.  **生物机制 (Mechanism_Target)**
    *   分子：靶点（受体/酶）、基因、蛋白质、生物标志物、信号通路、疾病等。
    *   机制：MoA、激动剂/拮抗剂、IC50/EC50、酶抑制、药代动力学参数以及值等。
4.  **临床与统计 (Clinical_Stats)**
    *   设计：分期(Phase I-IV)、双盲/随机、入排标准等。
    *   指标：终点(OS/PFS)、药代参数(Cmax/AUC/t1/2)等。
    *   统计：CI、HR、P值等。
5.  **疾病与安全 (Disease_Safety)**
    *   临床：适应症、疾病亚型、联合用药等。
    *   安全：不良反应(AE)、黑框警告、DDI等。

# Constraints (约束条件)
1. **格式严格**：输出必须且只能是一个 Python 风格的 **JSON** 格式。
2. **JSON结构定义**：必须包含一个核心的键 named_entities（实体列表）
    *   `named_entities`: 包含所有提取出的去重实体列表， 实体不能太长，一定要具有实体的特征，必须是名词或者名词短语
4. **严格去噪**：去除泛指词（如“患者”、“研究”），保留专有名词或名词短语
5. **精准提取**：
    *   实体：必须提取文本中原样出现的词汇，严禁翻译、简写转换或凭空捏造，文本中可能含有多种语言，对于多种语言的实体都需要提取出来。
6. **禁止废话**：不要输出任何 “好的”、“以下是结果” 等前言或后语，仅返回 JSON 字符串。
7. **列表纯净度**：两个列表均需去重，且不包含任何解释性文字、Markdown 格式标记或无关内容。
8. **不能出现给定文本中未提及的词汇**：提取文本中实际出现的实体，不能凭空添加任何不是给定文本中的内容。


# Output Schema (输出结构定义)
你必须返回一个符合 JSON 标准的对象，包含以下两个字段：

```json
{
  "named_entities": [
    "实体名称1",
    "实体名称2"
    // 注意：列表需去重，仅包含字符串，不能太长
  ]
}
```

"""


SYSTEM_PROMPT_TEMPLATE_IE = """
# Role
你是一名资深的严格的生物医药知识图谱构建专家，专注于仿制药开发、临床评价及转化医学领域。你精通从非结构化文本中进行命名实体识别（NER）和关系抽取（RE）。
，禁止根据常识推理文中未提及的信息。禁止修正文中的错别字。禁止输出文中不存在的词汇。

# Task
读取用户提供的【待分析文本】，执行以下操作：
在做实体抽取和关系构建之间，请先分析句子结构，
1.  **实体抽取**：识别文本中具有高价值的生物医药实体，必须是名词或者名词短语
2.  **关系构建**：基于上下文语义，提取实体之间的逻辑关系，构建三元组。
3.  **格式输出**：输出严格的 JSON 格式数据。

# 实体抽取标准：
请在抽取时严格参考以下分类体系：
1.  **药物与CMC (Drug_CMC)**
    *   物质：通用名、商品名、化学名、CAS号、API、辅料、杂质、代谢产物等。
    *   属性：晶型、盐型、溶解度、BCS分类等。
    *   工艺：剂型、给药途径、规格等。
2.  **法规与评价 (Regulatory_Evaluation)**
    *   核心：参比制剂(RLD)、生物等效性(BE)、一致性评价、治疗等效性(TE)、专利挑战(P4)等。
    *   法规：ANDA、DMF、橙皮书、CDE、MAH等。
3.  **生物机制 (Mechanism_Target)**
    *   分子：靶点（受体/酶）、基因、蛋白质、生物标志物、信号通路、疾病等。
    *   机制：MoA、激动剂/拮抗剂、IC50/EC50、酶抑制、药代动力学参数以及值等。
4.  **临床与统计 (Clinical_Stats)**
    *   设计：分期(Phase I-IV)、双盲/随机、入排标准等。
    *   指标：终点(OS/PFS)、药代参数(Cmax/AUC/t1/2)等。
    *   统计：CI、HR、P值等。
5.  **疾病与安全 (Disease_Safety)**
    *   临床：适应症、疾病亚型、联合用药等。
    *   安全：不良反应(AE)、黑框警告、DDI等。

# 关系构建标准：
在提取到实体的基础上，提取文本中明确表述的实体关系，形成三元组 `[主体, 关系, 客体]`。

# Constraints (约束条件)
1. **格式严格**：输出必须且只能是一个 Python 风格的 **JSON** 格式。
2. **JSON结构定义**：必须包含两个核心的键 named_entities（实体列表）、triple_relations（三元组列表）
    *   `named_entities`: 包含所有提取出的去重实体列表， 实体不能太长，一定要具有实体的特征，必须是名词或名词短语。
    *   `triple_relations`: 包含所有关系三元组的列表，每个三元组格式为 `["主体", "关系谓语", "客体"]`。
3. **关系逻辑约束**：
    *   三元组中的“主体”和“客体”**必须**存在于 `named_entities` 列表中。
    *   “关系谓语”应尽量简练且准确（如“治疗”、“抑制”、“包含”、“导致”），如果原文有明确动词，优先使用原文动词。
4. **严格去噪**：去除泛指词（如“患者”、“研究”），保留专有名词或名词短语。
5. **精准提取**：
    *   实体：必须提取文本中原样出现的词汇，严禁翻译、简写转换或凭空捏造，文本中可能含有多种语言，对于多种语言的实体都需要提取出来。
    *   三元组：“实体 1” 和 “实体 2” 必须是named_entities列表中已存在的实体；“关系” 需基于文本语义精准概括（可使用文本中出现的关系词或隐含语义的简洁表述，如 “用于治疗”“作用于”“对应” 等）。
6. **禁止废话**：不要输出任何 “好的”、“以下是结果” 等前言或后语，仅返回 JSON 字符串。
7. **列表纯净度**：两个列表均需去重，且不包含任何解释性文字、Markdown 格式标记或无关内容。
8. **不能出现给定文本中未提及的词汇**：提取文本中实际出现的实体和实体间关系，不能凭空添加任何不是给定文本中的内容。


# Output Schema (输出结构定义)
你必须返回一个符合 JSON 标准的对象，包含以下两个字段：

```json
{
  "named_entities": [
    "实体名称1",
    "实体名称2"
    // 注意：列表需去重，仅包含字符串，不能太长
  ],
  "triple_relations": [
    ["主体实体", "关系谓语", "客体实体"],
    ["主体实体", "关系谓语", "客体实体"]
    // 注意：主体和客体必须在 named_entities 中存在
  ]
}
```

"""


DICT_TASK = {
    "ner&re": SYSTEM_PROMPT_TEMPLATE_IE,
    "ner": SYSTEM_PROMPT_TEMPLATE_NER,
}

async def fetch_openai_ner(client: AsyncOpenAI, chunk_data, semaphore):
    """
    请求 API 并解析结果，统计 Token 和 NER/RE 数量
    """
    messages = [
        {"role": "system", "content": str(DICT_TASK[TASK])},
        {"role": "user", "content": str(chunk_data.get("content", ""))}
    ]
    
    # 初始化统计变量
    input_tokens = 0
    output_tokens = 0
    ner_num = 0
    re_num = 0
    extracted_entities = {}
    
    async with semaphore:
        start_time = time.time()
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                # max_tokens=8192,
                # top_p=0.1,
                # presence_penalty = 0.0,
                # frequency_penalty = 0.0,
                # extra_body={
                #     "top_k": 2, 
                #     "min_p": 0,
                #     "repetition_penalty": 1.0
                # },
            )
            
            # 1. 获取 Token 统计 (Ollama/OpenAI 兼容)
            # logger.info(response.json())
            if response.usage:
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
            
            # 2. 获取内容
            llm_content = response.choices[0].message.content
            llm_content = llm_content.replace("```json", "").replace("```", "")
            # 3. 解析 JSON 并统计实体/关系数量
            try:
                extracted_entities = json.loads(llm_content)
                
                # 统计数量 (确保类型安全)
                if isinstance(extracted_entities, dict):
                    entities = extracted_entities.get("named_entities", [])
                    relations = extracted_entities.get("triple_relations", [])
                    
                    if isinstance(entities, list):
                        ner_num = len(entities)
                    if isinstance(relations, list):
                        re_num = len(relations)
                        
            except json.JSONDecodeError:
                logger.warning(f"JSON Parse Error in chunk {chunk_data.get('chunk_id')}")
                extracted_entities = {"raw_text": llm_content, "error": "json_parse_fail"}

        except APITimeoutError:
            logger.error(f"Timeout processing chunk {chunk_data.get('chunk_id')}")
            extracted_entities = {"error": "timeout"}
        except APIError as e:
            logger.error(f"OpenAI API Error processing chunk {chunk_data.get('chunk_id')}: {e}")
            extracted_entities = {"error": "api_error", "details": str(e)}
        except Exception as e:
            logger.error(f"Unexpected Error processing chunk {chunk_data.get('chunk_id')}: {e}")
            extracted_entities = {"error": "exception", "details": str(e)}
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 4. 构造包含统计信息的结果
        result_record = {
            "No.": NUM,
            "model": MODEL_NAME,
            "task": TASK,
            "chunk_id": chunk_data.get("chunk_id"),
            "file_name": chunk_data.get("file_name"),
            "ner_result": extracted_entities,
            "original_content": chunk_data.get("content"),
            "processing_time_seconds": round(duration, 4),
            "timestamp": datetime.now().isoformat(),
            # 新增字段
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ner_num": ner_num,
            "re_num": re_num
        }
        
        logger.info(f"[{chunk_data.get('file_name')} - ID:{chunk_data.get('chunk_id')}] Done in {duration:.2f}s | Tokens: {input_tokens}/{output_tokens} | NER: {ner_num}, RE: {re_num}")
        return result_record

async def process_and_save_chunk(client, chunk, semaphore, output_path, file_lock):
    """
    请求完成后 -> 立即获取锁 -> 立即追加写入文件 -> 强制刷盘
    """
    # 1. 等待 API 结果
    result = await fetch_openai_ner(client, chunk, semaphore)
    
    # 2. 写入文件
    async with file_lock:
        try:
            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno()) 
        except Exception as e:
            logger.error(f"Failed to save chunk {chunk.get('chunk_id')}: {e}")

async def process_file(file_path, client, semaphore):
    """
    处理单个 JSON 文件
    """
    file_name = os.path.basename(file_path)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            chunks = [data]
        elif isinstance(data, list):
            chunks = data
        else:
            logger.info(f"Skipping {file_name}: Unknown format")
            return

        rel_path = os.path.relpath(file_path, INPUT_DIR)
        rel_dir = os.path.dirname(rel_path)
        target_dir = os.path.join(OUTPUT_DIR, rel_dir)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)
        
        # 使用 .jsonl 格式
        output_filename = f"{NUM}_{MODEL_NAME}_{TASK}_{file_name}"
        if output_filename.endswith('.json'):
            output_filename = output_filename[:-5] + '.jsonl'
        else:
            output_filename += '.jsonl'
             
        output_path = os.path.join(target_dir, output_filename)
        
        # 初始化文件（清空）
        with open(output_path, 'w', encoding='utf-8') as f:
            pass 

        file_lock = asyncio.Lock()

        tasks = [
            process_and_save_chunk(client, chunk, semaphore, output_path, file_lock) 
            for chunk in chunks
        ]
        
        await asyncio.gather(*tasks)
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

async def main():
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    search_pattern = os.path.join(INPUT_DIR, "**/*.json")
    json_files = glob.glob(search_pattern, recursive=True)
    logger.info(f"Found {len(json_files)} files in {INPUT_DIR}...")

    client = AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_API_URL,
        timeout=6000, 
        max_retries=1
    )

    try:
        tasks = [process_file(f, client, semaphore) for f in json_files]
        
        overall_start = time.time()
        await asyncio.gather(*tasks)
        overall_end = time.time()
        
        logger.info(f"\nAll tasks finished! Total time: {overall_end - overall_start:.2f}s")
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())

