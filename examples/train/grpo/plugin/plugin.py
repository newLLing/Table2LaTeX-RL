import asyncio
import re
from typing import List
import json
from table_recognition_metric import TEDS
from swift.plugin import ORM, orms
from swift.utils import get_logger
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Tuple
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import subprocess
logger = get_logger()
REWARD_TEMP_DIR = "reward_temp"
os.makedirs(REWARD_TEMP_DIR, exist_ok=True)

def remove_grid_lines(latex_table):
    # 去除 \hline, \cline, \toprule, \midrule, \bottomrule
    cleaned_table = re.sub(r'\\cmidrule{\s*}|\\cdashline\{[0-9]+(-[0-9]+)?\}\s*|\\cmidrule\((?:lr|r|l)?\)\{[0-9]+\-[0-9]+\}\s*|\\arrayrulecolor{.*?}\s*|\\caption{.*?}\s*|\\centering\s*|\\hline\s*|\\cline{.*?}\s*|\\toprule\s*|\\midrule\s*|\\bottomrule\s*', '', latex_table)
    
    cleaned_table = re.sub(r'\\tabularnewline', r'\\\\', cleaned_table)
    # # 去除注释
    # cleaned_table = re.sub(r'(?<!\\)%.*$', '', cleaned_table, flags=re.MULTILINE)
    # cleaned_table = re.sub(r'(?<!\\)\[[^\[\]]*\]', '', cleaned_table)
    # 合并连续的空行
    cleaned_table = re.sub(r'\n\s*\n', '\n', cleaned_table)
    
    return cleaned_table.strip(' \n')  # 去除首尾空格

def fix_multi(cell):
    multirow_pattern = r'\\multirow{(\d+)}{.*?}{(.*?)}'
    multicol_pattern = r'\\multicolumn{(\d+)}{.*?}{(.*?)}'
    
    match = re.search(multirow_pattern, cell['content'])
    if match:
        cell['rowspan'] = int(match.group(1))
        cell['content'] = cell['content'].replace(match.group(0), match.group(2).strip(), 1).strip()

    match = re.search(multicol_pattern, cell['content'])
    if match:
        cell['colspan'] = int(match.group(1))
        cell['content'] = cell['content'].replace(match.group(0), match.group(2).strip(), 1).strip()

    return cell    

def grid2html(grid):
    def to_td(grid, r, c):
        if grid[r][c] == '<<' or grid[r][c] == '^^' or grid[r][c] == '..':
            return ''
        td = {'text': grid[r][c], 'rowspan':1, 'colspan': 1}

        for i in range(r + 1, len(grid)):
            if grid[i][c] == '^^':
                td['rowspan'] += 1
            else:
                break
        
        for j in range(c + 1, len(grid[r])):
            if grid[r][j] == '<<':
                td['colspan'] += 1
            else:
                break
        return f'<td rowspan={td["rowspan"]} colspan={td["colspan"]}> {td["text"]} </td>'.replace('rowspan=1', '').replace('colspan=1', '')
        
    
    html = []
    for r in range(len(grid)):
        row = []
        for c in range(len(grid[0])):
            row.append(to_td(grid, r, c))
        html.append(f'<tr> {"".join(row)} </tr>')
    # for row in grid:
    #     html.append('<tr>' + ''.join([to_td(c) for c in row]) + '</tr>')
    
    return '<html><body><table>' + '\n'.join(html) + '</table></body></html>'


def qylatex_to_grid(latex):
    if not re.search(r'\\end{tabular[x]*\*?\}', latex):
        return
    pattern = r'\\begin\{tabular[x]*\*?\}.*?\\end\{tabular[x]*\*?\}'
    matches = re.findall(pattern, latex, re.DOTALL)
    if not matches:
        return
    content = remove_grid_lines(matches[0])
    rows = content.strip(' \n').split(r'\\')
    processed_rows = []
    for row in rows:
        if not row.strip():
            continue
        columns = re.split(r'(?<!\\)&', row)
        columns = [fix_multi({'content': c.strip(' \n'), 'rowspan': 1, 'colspan': 1}) for c in columns]
        processed_rows.append(columns)
    max_cols = max([sum([it['colspan'] for it in r]) for r in processed_rows]) if processed_rows else 0
    grid = [[None for _ in range(max_cols)] for _ in range(len(processed_rows))]
    r_idx_bias = 0
    for r_idx, row in enumerate(processed_rows):
        r_idx += r_idx_bias
        while r_idx >= len(grid):
            grid.append([None for _ in range(max_cols)])
        c_idx = 0
        current_row_bias = 10000
        for cell in row:
            while c_idx < len(grid[r_idx]) and grid[r_idx][c_idx] is not None:
                c_idx += 1
            if c_idx >= len(grid[r_idx]):
                break
            current_row_bias = min(current_row_bias, cell['rowspan'])
            grid[r_idx][c_idx] = cell['content']
            for r in range(cell['rowspan']):
                for c in range(cell['colspan']):
                    if r == 0 and c == 0:
                        continue
                    target_r = r_idx + r
                    target_c = c_idx + c
                    if target_r >= len(grid):
                        grid.append([None for _ in range(max_cols)])
                    if target_c < len(grid[target_r]):
                        grid[target_r][target_c] = '^^' if r > 0 else ('<<' if c > 0 else '..')
            c_idx += cell['colspan']
        r_idx_bias += current_row_bias - 1
    grid = [[c if c is not None else '' for c in r] for r in grid]
    return grid


def latex2html(latex_str):
    # 去除注释
    latex_str = re.sub(r'(?<!\\)%.*$', '', latex_str, flags=re.MULTILINE)
    # 去除"\\\\[...]"
    latex_str = re.sub(r'(?<!\\)\\\\\[.*?\]', '', latex_str, flags=re.DOTALL)

    latex_str = latex_str.replace('\n', '').replace('\t', '')
    try:
        grid = qylatex_to_grid(latex_str)
    except IndexError:
        return 
    if not grid:
        return
    html = grid2html(grid)
    return html


def init_worker():
    """子进程初始化函数"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
def cleanup_temp_files(temp_dir: str) -> None:
    """清理临时文件"""
    extensions = ['.tex', '.log', '.aux']
    for ext in extensions:
        temp_file = os.path.join(temp_dir, f"temp{ext}")
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                logger.error(f"删除临时文件{temp_file}失败: {str(e)}")
def safe_render_tex(content: str, timeout: int = 10) -> str:
    """安全渲染LaTeX到PDF,返回保留的PDF路径"""
    temp_id = uuid.uuid4().hex[:8]
    temp_dir = os.path.join("reward_temp", temp_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    tex_path = os.path.join(temp_dir, "temp.tex")
    pdf_path = os.path.join(temp_dir, "temp.pdf")
   
    try:
        latex_template = (
            r"\documentclass[standalone]{article}"
            r"\usepackage[utf8]{inputenc}"
            r"\usepackage[T1]{fontenc}"
            r"\usepackage{amsmath, amsthm, amssymb, graphicx, geometry, array,xcolor}"
            r"\usepackage{booktabs, multirow, natbib, tabularx, multicol, bm}"
            r"\begin{document}"
            r"\begin{table}"
            r"\centering"
            r"\resizebox{0.5\columnwidth}{!}{"
            r"%s"  # 插入表格内容
            r"}"
            r"\end{table}"
            r"\end{document}"
        ) % content

        with open(tex_path, 'w') as f:
            f.write(latex_template)

        # 执行pdflatex
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", temp_dir, tex_path],
            check=False,
            capture_output=True,
            text=True,
            encoding='latin-1',  # Decode output as Latin-1
            timeout=timeout
        )
        cleanup_temp_files(temp_dir)  # 清理临时文件
        # 验证编译结果
        if not os.path.exists(pdf_path):
            logger.error(f"LaTeX编译失败:{result.stderr.decode()}")
            return ""

        return pdf_path

    except Exception as e:
        logger.error(f"渲染异常：{str(e)}")
        return ""



def safe_convert_pdf_to_png(pdf_path: str, dpi: int = 300) -> str:
    """安全转换PDF到PNG,返回PNG路径"""
    if not os.path.exists(pdf_path):
        return ""
    
    temp_dir = os.path.join("reward_temp", uuid.uuid4().hex[:8])
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        images = convert_from_path(
            pdf_path, 
            dpi=dpi, 
            output_folder=temp_dir,
            poppler_path="/usr/bin"
        )
        if not images:
            return ""
            
        png_path = os.path.join(temp_dir, "output.png")
        images[0].save(png_path, "PNG")
        return png_path
        
    except Exception as e:
        logger.error(f"PNG转换失败:{str(e)}")
        return ""
    
def calculate_similarity(img1_path: str, img2_path: str) -> float:
    try:
        # 调用CW-SSIM计算相似度
        return calculate_cwssim(img1_path, img2_path)
    except Exception as e:
        logger.error(f"相似度计算失败：{str(e)}")
        return 0.0

def process_accuracy(comp, sol):
    comp_pdf, sol_pdf, comp_png, sol_png = None, None, None, None
    try:
        # 生成PDF
        comp_pdf = safe_render_tex(comp)
        sol_pdf = safe_render_tex(sol)
        
        if not (comp_pdf and sol_pdf):
            return 0.0
            
        # 转换PNG
        comp_png = safe_convert_pdf_to_png(comp_pdf)
        sol_png = safe_convert_pdf_to_png(sol_pdf)
        
        # 计算相似度
        similarity = calculate_similarity(comp_png, sol_png) if comp_png and sol_png else 0
        print(f"Using GPU ID: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
        print(f"cwssim: {similarity}")
        reward = 1.0 if similarity>0.6 else 0.0
        for path in [comp_pdf, sol_pdf, comp_png, sol_png]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        return reward
    except Exception as e:
        # 异常时清理生成的文件
        for path in [comp_pdf, sol_pdf, comp_png, sol_png]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        return 0.0
def process_format(content,sol):
    """格式正确性奖励"""
    try:
        teds_structure_score = teds_structure(sol, content)
        
        print(f"teds_structure_score: {teds_structure_score}")
        reward = 1.0 if teds_structure_score>0.9 else 0.0
        return reward
    except Exception as e:
        return 0.0

def teds_structure(gt, pred):
    """计算结构相似度"""
    gt_html = latex2html(gt)
    pred_html = latex2html(pred)
    if not pred_html:
        # print("Prediction LaTeX to HTML conversion failed.")
        return 0
    structure_teds = TEDS(structure_only=True)
    structure_score = structure_teds(gt_html, pred_html)

    return structure_score

def dwt2_simple(img_array):
    """Performs a simple 2D discrete wavelet transform with even-size check"""
    rows, cols = img_array.shape
    rows = rows - rows % 2
    cols = cols - cols % 2
    img_array = img_array[:rows, :cols]
    
    blocks = img_array.reshape(rows//2, 2, cols//2, 2).transpose(0, 2, 1, 3)
    a = blocks[..., 0, 0]
    b = blocks[..., 0, 1]
    c = blocks[..., 1, 0]
    d = blocks[..., 1, 1]
    
    cA = (a + b + c + d) * 0.25
    cH = (a - c) * 0.5
    cV = (a - b) * 0.5
    cD = (a - d) * 0.5
    return cA, cH, cV, cD

def calculate_ssim(img1, img2):
    """针对黑白表格图像优化的SSIM计算"""
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()
    n = img1_flat.size
    
    # 计算均值
    mean1 = np.mean(img1_flat)
    mean2 = np.mean(img2_flat)
    
    # 计算方差和协方差
    var1 = np.var(img1_flat)
    var2 = np.var(img2_flat)
    covar = np.cov(img1_flat, img2_flat)[0,1]
    
    # 针对黑白图像优化的常量
    L = 255.0  # 像素最大值
    k1 = 0.01  # 推荐值
    k2 = 0.03  # 推荐值
    
    # 计算稳定常量
    C1 = (k1 * L) ** 2  # 亮度比较的稳定常量
    C2 = (k2 * L) ** 2  # 对比度比较的稳定常量
    
    # SSIM计算
    numerator = (2 * mean1 * mean2 + C1) * (2 * covar + C2)
    denominator = (mean1**2 + mean2**2 + C1) * (var1 + var2 + C2)
    
    ssim = numerator / denominator
    
    # 对于完全空白的图像进行特殊处理
    if np.std(img1_flat) < 1e-6 and np.std(img2_flat) < 1e-6:
        if np.mean(img1_flat) == np.mean(img2_flat):
            return 1.0
        else:
            return 0.0
            
    return max(0.0, min(1.0, ssim))  # 确保结果在[0,1]范围内

def calculate_cwssim(image1_path, image2_path):
    """Compares two images using Complex Wavelet Structural Similarity Index (CW-SSIM)"""
    try:
        image1 = Image.open(image1_path).convert('L')
        image2 = Image.open(image2_path).convert('L')
        
        image1 = image1.resize(image2.size)
        
        img1_array = np.array(image1)
        img2_array = np.array(image2)
        
        cA1, cH1, cV1, cD1 = dwt2_simple(img1_array)
        cA2, cH2, cV2, cD2 = dwt2_simple(img2_array)
        
        ssim_cA = calculate_ssim(cA1, cA2)
        ssim_cH = calculate_ssim(cH1, cH2)
        ssim_cV = calculate_ssim(cV1, cV2)
        ssim_cD = calculate_ssim(cD1, cD2)
        cwssim_score = (ssim_cA + ssim_cH + ssim_cV + ssim_cD) / 4
        
        return cwssim_score
    
    except Exception as e:
        print(f"Error comparing images {image1_path} and {image2_path}: {e}")
        return 0.0

# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards



class Table2LatexAcc(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
 
        print(f"completions' length: {len(completions)}")

        try:
            results = []
            result = process_accuracy(completions[0], solution[0]['content'])
            results.append(result)
            return results
        except Exception as e:
            logger.error(f"准确性奖励计算失败: {str(e)}")
            return [0.0] * len(completions)

class Table2Latexform(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        
        """格式正确性奖励"""
        try:
            results = []
            result = process_format(completions[0], solution[0]['content'])
            results.append(result)
            return results
        except Exception as e:
            logger.error(f"格式奖励计算失败: {str(e)}")
            return [0.0] * len(completions)

orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_code_reward'] = CodeReward
orms['external_code_format'] = CodeFormat
orms['external_table2latex_acc'] = Table2LatexAcc
orms['external_table2latex_form'] = Table2Latexform
