import asyncio
import re
from typing import List, Tuple, Optional
import json
from table_recognition_metric import TEDS
from swift.plugin import ORM, orms
from swift.utils import get_logger
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import subprocess
import hashlib
import atexit
import shutil
from datetime import datetime, timedelta


logger = get_logger()
REWARD_TEMP_DIR = "reward_temp"
os.makedirs(REWARD_TEMP_DIR, exist_ok=True)

# ============================================================================
# 预编译正则表达式
# ============================================================================
PATTERN_GRID_LINES = re.compile(
    r'\\cmidrule{\s*}|\\cdashline\{[0-9]+(-[0-9]+)?\}\s*|'
    r'\\cmidrule\((?:lr|r|l)?\)\{[0-9]+\-[0-9]+\}\s*|'
    r'\\arrayrulecolor{.*?}\s*|\\caption{.*?}\s*|\\centering\s*|'
    r'\\hline\s*|\\cline{.*?}\s*|\\toprule\s*|\\midrule\s*|\\bottomrule\s*'
)
PATTERN_TABULAR = re.compile(r'\\begin\{tabular[x]*\*?\}.*?\\end\{tabular[x]*\*?\}', re.DOTALL)
PATTERN_TABULARNEWLINE = re.compile(r'\\tabularnewline')
PATTERN_NEWLINES = re.compile(r'\n\s*\n')
PATTERN_COMMENTS = re.compile(r'(?<!\\)%.*$', re.MULTILINE)
PATTERN_BRACKET_CONTENT = re.compile(r'(?<!\\)\\\\\[.*?\]', re.DOTALL)
PATTERN_MULTIROW = re.compile(r'\\multirow{(\d+)}{.*?}{(.*?)}')
PATTERN_MULTICOL = re.compile(r'\\multicolumn{(\d+)}{.*?}{(.*?)}')

# ============================================================================
# 临时文件管理器
# ============================================================================
# ------------------ 修复后的全局缓存与 TempFileManager 实现 ------------------

# 全局渲染缓存（内存缓存） — 提前定义，保证类和回调使用时可见
_render_cache = {}
_cache_lock = threading.Lock()

class TempFileManager:
    """统一管理临时文件/目录：
       - 提供 get_temp_dir() 创建独立子目录
       - safe_remove(path) 安全删除文件
       - 后台守护线程周期性调用 cleanup_old_files()
       - shutdown() 在退出时停止线程并做一次清理
    """
    def __init__(self, base_dir="reward_temp", max_age_hours=1, max_cache_size_mb=500, cleanup_interval_s=600):
        self.base_dir = base_dir
        self.max_age = timedelta(hours=max_age_hours)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # bytes
        self.cleanup_interval = cleanup_interval_s
        os.makedirs(self.base_dir, exist_ok=True)

        # 用于管理临时目录的集合（可根据需要扩展）
        self._active_dirs = set()
        self._stop_event = threading.Event()
        self._cleanup_thread = None

        # 在退出时优雅清理：先停止线程再调用一次清理
        atexit.register(self.shutdown)

        # 启动后台清理线程（守护线程）
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """启动周期性清理线程（守护线程）"""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return

        def _loop():
            while not self._stop_event.wait(self.cleanup_interval):
                try:
                    # 每次循环调用清理函数
                    self.cleanup_old_files()
                except Exception as e:
                    logger.debug(f"后台清理线程异常: {e}")

        self._cleanup_thread = threading.Thread(target=_loop, daemon=True, name="TempFileManager-Cleanup")
        self._cleanup_thread.start()
        logger.debug("临时文件管理后台清理线程已启动")

    def shutdown(self):
        """停止后台线程并做一次清理（用于 atexit）"""
        try:
            self._stop_event.set()
            if self._cleanup_thread is not None:
                self._cleanup_thread.join(timeout=5)
            # 最后一次清理
            self.cleanup_old_files()
            logger.debug("TempFileManager 已 shutdown 并完成一次清理")
        except Exception as e:
            logger.debug(f"shutdown 异常: {e}")

    def get_temp_dir(self) -> str:
        """返回一个唯一的临时子目录路径（确保存在），便于并发使用"""
        subdir = uuid.uuid4().hex
        dir_path = os.path.join(self.base_dir, subdir)
        try:
            os.makedirs(dir_path, exist_ok=True)
            # 记录以便调试/管理（不是必须）
            self._active_dirs.add(dir_path)
        except Exception as e:
            logger.debug(f"创建临时目录失败 {dir_path}: {e}")
            # 退回到 base_dir（尽量保证不会返回不存在路径）
            dir_path = self.base_dir
        return dir_path

    def safe_remove(self, path: str):
        """安全删除文件或空目录，静默失败"""
        try:
            if not path:
                return
            if os.path.isdir(path):
                # 仅删除空目录；若需要强制删除请使用 shutil.rmtree
                try:
                    os.rmdir(path)
                except OSError:
                    # 非空或无法删除，则尝试递归删除（基于调用方意图）
                    shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.debug(f"safe_remove 失败 {path}: {e}")

    def get_cache_size(self) -> int:
        """获取缓存目录的总大小（保留原实现）"""
        total_size = 0
        try:
            for dirname in os.listdir(self.base_dir):
                dir_path = os.path.join(self.base_dir, dirname)
                if os.path.isdir(dir_path):
                    total_size += sum(
                        os.path.getsize(os.path.join(dir_path, f))
                        for f in os.listdir(dir_path)
                        if os.path.isfile(os.path.join(dir_path, f))
                    )
        except Exception as e:
            logger.debug(f"计算缓存大小失败: {e}")
        return total_size

    def cleanup_old_files(self):
        """清理超过时间限制或超过大小限制的临时文件/目录"""
        try:
            now = datetime.now()

            # 收集所有目录及其创建时间与大小
            dirs_with_time = []
            for dirname in os.listdir(self.base_dir):
                dir_path = os.path.join(self.base_dir, dirname)
                if os.path.isdir(dir_path):
                    try:
                        create_time = datetime.fromtimestamp(os.path.getctime(dir_path))
                        
                        # ⭐ 改进：递归计算目录大小（包括所有子文件）
                        dir_size = 0
                        for root, _, files in os.walk(dir_path):
                            for f in files:
                                file_path = os.path.join(root, f)
                                try:
                                    dir_size += os.path.getsize(file_path)
                                except:
                                    pass
                        
                        dirs_with_time.append((dir_path, create_time, dir_size))
                    except Exception as e:
                        logger.debug(f"获取目录信息失败 {dir_path}: {e}")

            # 按创建时间排序（最旧的在前）
            dirs_with_time.sort(key=lambda x: x[1])

            cleaned_count = 0
            total_size = sum(d[2] for d in dirs_with_time)

            for dir_path, create_time, dir_size in dirs_with_time:
                should_delete = False
                reason = None

                # 条件1: 超过最大年龄
                if now - create_time > self.max_age:
                    should_delete = True
                    reason = "超时"

                # 条件2: 总大小超限（删除最旧目录）
                elif total_size > self.max_cache_size:
                    should_delete = True
                    reason = "缓存超限"

                if should_delete:
                    # 从全局渲染缓存中移除对应条目（如果存在）
                    try:
                        with _cache_lock:
                            to_remove = [k for k, v in _render_cache.items() 
                                       if isinstance(v, str) and v.startswith(dir_path)]
                            for key in to_remove:
                                del _render_cache[key]
                    except Exception as e:
                        logger.debug(f"从 _render_cache 移除失败: {e}")

                    # ⭐ 改进：强制删除目录（包括所有内容）
                    try:
                        # 先尝试删除目录中的所有 .ppm 文件
                        for root, _, files in os.walk(dir_path):
                            for file in files:
                                if file.endswith('.ppm'):
                                    try:
                                        os.remove(os.path.join(root, file))
                                    except:
                                        pass
                        
                        # 然后强制删除整个目录
                        shutil.rmtree(dir_path, ignore_errors=False)
                        self._active_dirs.discard(dir_path)
                        cleaned_count += 1
                        total_size -= dir_size
                        logger.debug(f"清理 {dir_path} ({reason})")
                    except Exception as e:
                        logger.error(f"删除目录失败 {dir_path}: {e}")
                        # ⭐ 如果失败，尝试强制删除（ignore_errors=True）
                        try:
                            shutil.rmtree(dir_path, ignore_errors=True)
                        except:
                            pass

            if cleaned_count > 0:
                logger.info(f"清理了 {cleaned_count} 个临时目录，当前缓存大小: {total_size/1024/1024:.2f} MB")

        except Exception as e:
            logger.error(f"临时文件清理失败: {e}")


# 创建单例 temp_manager（与原来行为一致）
temp_manager = TempFileManager(
    base_dir=REWARD_TEMP_DIR,
    max_age_hours=1,          # 1小时后清理
    max_cache_size_mb=500,    # 最大缓存500MB
    cleanup_interval_s=600    # 后台线程每10分钟清理一次
)

# ============================================================================
# LaTeX处理函数（使用预编译正则）
# ============================================================================
def remove_grid_lines(latex_table: str) -> str:
    """去除LaTeX表格中的网格线"""
    cleaned = PATTERN_GRID_LINES.sub('', latex_table)
    cleaned = PATTERN_TABULARNEWLINE.sub(r'\\\\', cleaned)
    cleaned = PATTERN_NEWLINES.sub('\n', cleaned)
    return cleaned.strip(' \n')

def fix_multi(cell: dict) -> dict:
    """处理multirow和multicolumn"""
    match = PATTERN_MULTIROW.search(cell['content'])
    if match:
        cell['rowspan'] = int(match.group(1))
        cell['content'] = cell['content'].replace(match.group(0), match.group(2).strip(), 1).strip()

    match = PATTERN_MULTICOL.search(cell['content'])
    if match:
        cell['colspan'] = int(match.group(1))
        cell['content'] = cell['content'].replace(match.group(0), match.group(2).strip(), 1).strip()

    return cell    

def grid2html(grid: List[List[str]]) -> str:
    """将grid转换为HTML表格"""
    def to_td(grid, r, c):
        if grid[r][c] in ('<<', '^^', '..'):
            return ''
        td = {'text': grid[r][c], 'rowspan': 1, 'colspan': 1}

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
        
        rowspan_str = f'rowspan={td["rowspan"]}' if td['rowspan'] > 1 else ''
        colspan_str = f'colspan={td["colspan"]}' if td['colspan'] > 1 else ''
        attrs = ' '.join(filter(None, [rowspan_str, colspan_str]))
        return f'<td {attrs}> {td["text"]} </td>'.strip()
    
    html = []
    for r in range(len(grid)):
        row = []
        for c in range(len(grid[0])):
            td_html = to_td(grid, r, c)
            if td_html:
                row.append(td_html)
        html.append(f'<tr> {"".join(row)} </tr>')
    
    return '<html><body><table>' + '\n'.join(html) + '</table></body></html>'

def qylatex_to_grid(latex: str) -> Optional[List[List[str]]]:
    """将LaTeX表格转换为grid格式"""
    if not re.search(r'\\end{tabular[x]*\*?\}', latex):
        return None
    
    matches = PATTERN_TABULAR.findall(latex)
    if not matches:
        return None
    
    content = remove_grid_lines(matches[0])
    rows = content.strip(' \n').split(r'\\')
    processed_rows = []
    
    for row in rows:
        if not row.strip():
            continue
        columns = re.split(r'(?<!\\)&', row)
        columns = [fix_multi({'content': c.strip(' \n'), 'rowspan': 1, 'colspan': 1}) for c in columns]
        processed_rows.append(columns)
    
    if not processed_rows:
        return None
        
    max_cols = max([sum([it['colspan'] for it in r]) for r in processed_rows])
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

def latex2html(latex_str: str) -> Optional[str]:
    """LaTeX转HTML"""
    latex_str = PATTERN_COMMENTS.sub('', latex_str)
    latex_str = PATTERN_BRACKET_CONTENT.sub('', latex_str)
    latex_str = latex_str.replace('\n', '').replace('\t', '')
    
    try:
        grid = qylatex_to_grid(latex_str)
    except IndexError:
        return None
    
    if not grid:
        return None
    
    html = grid2html(grid)
    return html

# ============================================================================
# 缓存的LaTeX渲染
# ============================================================================
@lru_cache(maxsize=2048)
def cached_latex_hash(content: str) -> str:
    """计算LaTeX内容的hash（用于缓存key）"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# 全局渲染缓存（内存缓存）
_render_cache = {}
_cache_lock = threading.Lock()

def cleanup_temp_files(temp_dir: str) -> None:
    """清理LaTeX编译的临时文件"""
    extensions = [
        '.log', '.aux', '.out',           # 基础文件
        '.fls', '.fdb_latexmk',            # latexmk
        '.synctex.gz',                     # SyncTeX
        '.toc', '.lof', '.lot',            # 目录
        '.bbl', '.blg',                    # BibTeX
        '.nav', '.snm', '.vrb'             # Beamer
    ]
    
    # 清理标准扩展名文件
    for ext in extensions:
        temp_file = os.path.join(temp_dir, f"temp{ext}")
        temp_manager.safe_remove(temp_file)
    
    # ⭐ 新增：清理所有 .ppm 文件（pdf2image 产生的中间文件）
    try:
        for file in os.listdir(temp_dir):
            if file.endswith('.ppm'):
                ppm_path = os.path.join(temp_dir, file)
                temp_manager.safe_remove(ppm_path)
                logger.debug(f"清理 .ppm 文件: {ppm_path}")
    except Exception as e:
        logger.debug(f"清理 .ppm 文件失败: {e}")
def safe_render_tex(content: str, timeout: int = 10) -> str:
    """安全渲染LaTeX到PDF，返回保留的PDF路径（带缓存）"""
    content_hash = cached_latex_hash(content)
    
    # 检查缓存
    with _cache_lock:
        if content_hash in _render_cache:
            cached_path = _render_cache[content_hash]
            if os.path.exists(cached_path):
                logger.debug(f"使用缓存的PDF: {cached_path}")
                return cached_path
            else:
                # 缓存失效，删除
                del _render_cache[content_hash]
    
    # 渲染新的PDF
    temp_dir = temp_manager.get_temp_dir()
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
            r"%s"
            r"}"
            r"\end{table}"
            r"\end{document}"
        ) % content

        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex_template)

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", temp_dir, tex_path],
            check=False,
            capture_output=True,
            text=True,
            encoding='latin-1',
            timeout=timeout
        )
        
        cleanup_temp_files(temp_dir)

        if not os.path.exists(pdf_path):
            logger.error(f"LaTeX编译失败: {result.stderr}")
            return ""

        # 加入缓存
        with _cache_lock:
            _render_cache[content_hash] = pdf_path

        return pdf_path

    except Exception as e:
        logger.error(f"渲染异常: {str(e)}")
        return ""

def safe_convert_pdf_to_png(pdf_path: str, dpi: int =300) -> str:
    """安全转换PDF到PNG（降低DPI以提升速度）"""
    if not os.path.exists(pdf_path):
        return ""
    
    temp_dir = temp_manager.get_temp_dir()
    
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
        
        # ⭐ 新增：立即清理 pdf2image 产生的临时 .ppm 文件
        cleanup_temp_files(temp_dir)
        
        return png_path
        
    except Exception as e:
        logger.error(f"PNG转换失败: {str(e)}")
        # ⭐ 新增：失败时也尝试清理
        cleanup_temp_files(temp_dir)
        return ""
# ============================================================================
# GPU加速的SSIM计算（可选）
# ============================================================================
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

def dwt2_simple(img_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """简单的2D离散小波变换"""
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

def calculate_ssim_cpu(img1: np.ndarray, img2: np.ndarray) -> float:
    """CPU版本的SSIM计算"""
    img1_flat = img1.ravel()
    img2_flat = img2.ravel()
    
    mean1 = np.mean(img1_flat)
    mean2 = np.mean(img2_flat)
    var1 = np.var(img1_flat)
    var2 = np.var(img2_flat)
    covar = np.cov(img1_flat, img2_flat)[0, 1]
    
    L = 255.0
    k1 = 0.01
    k2 = 0.03
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    
    numerator = (2 * mean1 * mean2 + C1) * (2 * covar + C2)
    denominator = (mean1**2 + mean2**2 + C1) * (var1 + var2 + C2)
    
    ssim = numerator / denominator
    
    # 处理空白图像
    if np.std(img1_flat) < 1e-6 and np.std(img2_flat) < 1e-6:
        return 1.0 if np.mean(img1_flat) == np.mean(img2_flat) else 0.0
            
    return max(0.0, min(1.0, ssim))

def calculate_ssim_gpu(img1: np.ndarray, img2: np.ndarray) -> float:
    """GPU加速的SSIM计算"""
    device = torch.device('cuda')
    
    img1_t = torch.from_numpy(img1).float().to(device)
    img2_t = torch.from_numpy(img2).float().to(device)
    
    mean1 = img1_t.mean()
    mean2 = img2_t.mean()
    var1 = img1_t.var()
    var2 = img2_t.var()
    
    img1_flat = img1_t.reshape(-1)
    img2_flat = img2_t.reshape(-1)
    covar = torch.mean((img1_flat - mean1) * (img2_flat - mean2))
    
    L, k1, k2 = 255.0, 0.01, 0.03
    C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
    
    numerator = (2 * mean1 * mean2 + C1) * (2 * covar + C2)
    denominator = (mean1**2 + mean2**2 + C1) * (var1 + var2 + C2)
    
    return torch.clamp(numerator / denominator, 0.0, 1.0).item()

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """自动选择CPU/GPU的SSIM计算"""
    if TORCH_AVAILABLE:
        try:
            return calculate_ssim_gpu(img1, img2)
        except Exception as e:
            logger.debug(f"GPU计算失败，回退到CPU: {e}")
    return calculate_ssim_cpu(img1, img2)

def calculate_cwssim(image1_path: str, image2_path: str) -> float:
    """使用复小波结构相似性指数比较两张图像"""
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
        logger.error(f"图像比较错误 {image1_path} 和 {image2_path}: {e}")
        return 0.0

def calculate_similarity(img1_path: str, img2_path: str) -> float:
    """计算图像相似度"""
    try:
        return calculate_cwssim(img1_path, img2_path)
    except Exception as e:
        logger.error(f"相似度计算失败: {str(e)}")
        return 0.0

# ============================================================================
# TEDS结构相似度
# ============================================================================
@lru_cache(maxsize=2048)
def teds_structure_cached(gt_hash: str, pred_hash: str, gt: str, pred: str) -> float:
    """带缓存的TEDS结构计算"""
    gt_html = latex2html(gt)
    pred_html = latex2html(pred)
    
    if not pred_html:
        return 0.0
    
    structure_teds = TEDS(structure_only=True)
    structure_score = structure_teds(gt_html, pred_html)
    return structure_score

def teds_structure(gt: str, pred: str) -> float:
    """计算结构相似度（带缓存）"""
    gt_hash = cached_latex_hash(gt)
    pred_hash = cached_latex_hash(pred)
    return teds_structure_cached(gt_hash, pred_hash, gt, pred)

# ============================================================================
# 处理函数（支持提前终止）
# ============================================================================
def process_accuracy(comp: str, sol: str) -> float:
    """准确性奖励（带提前终止优化）"""
    comp_pdf, sol_pdf, comp_png, sol_png = None, None, None, None
    try:
        # 提前终止：先做快速结构检查
        teds_score = teds_structure(sol, comp)
        logger.info(f"TEDS结构得分: {teds_score}")
        
        if teds_score < 0.3:  # 结构差异太大，直接返回0
            logger.info("结构差异过大，跳过视觉比较")
            return 0.0
        
        # 生成PDF
        comp_pdf = safe_render_tex(comp)
        sol_pdf = safe_render_tex(sol)
        
        if not (comp_pdf and sol_pdf):
            return 0.0
            
        # 转换PNG
        comp_png = safe_convert_pdf_to_png(comp_pdf)
        sol_png = safe_convert_pdf_to_png(sol_pdf)
        
        if not (comp_png and sol_png):
            return 0.0
        
        # 计算相似度
        similarity = calculate_similarity(comp_png, sol_png)
        logger.info(f"CW-SSIM: {similarity}")
        
        reward = 1.0 if similarity > 0.6 else 0.0
        
        # 清理临时PNG文件（PDF保留在缓存中）
        temp_manager.safe_remove(comp_png)
        temp_manager.safe_remove(sol_png)
        
        return reward
        
    except Exception as e:
        logger.error(f"准确性计算异常: {str(e)}")
        # 清理文件
        for path in [comp_png, sol_png]:
            temp_manager.safe_remove(path)
        return 0.0

def process_format(content: str, sol: str) -> float:
    """格式正确性奖励"""
    try:
        teds_structure_score = teds_structure(sol, content)
        logger.info(f"TEDS结构得分: {teds_structure_score}")
        reward = 1.0 if teds_structure_score > 0.9 else 0.0
        return reward
    except Exception as e:
        logger.error(f"格式计算异常: {str(e)}")
        return 0.0

# ============================================================================
# 多进程初始化
# ============================================================================
def init_worker():
    """子进程初始化函数"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# ORM类（支持批量并行处理）
# ============================================================================
class Table2LatexAcc(ORM):
    """准确性奖励（优化版：支持批量并行）"""
    
    def __init__(self):
        super().__init__()
        # 预创建进程池（避免重复创建销毁）
        self.pool = None
        self._pool_lock = threading.Lock()
    
    def _get_pool(self):
        """懒加载进程池"""
        if self.pool is None:
            with self._pool_lock:
                if self.pool is None:
                    num_workers = min(cpu_count(), 8)
                    self.pool = Pool(processes=num_workers, initializer=init_worker)
                    logger.info(f"创建进程池，worker数量: {num_workers}")
        return self.pool
    
    def __call__(self, completions: List[str], solution: List[dict], **kwargs) -> List[float]:
        """
        批量计算奖励
        Args:
            completions: 生成的输出列表
            solution: 标准答案列表
        Returns:
            奖励分数列表
        """
        logger.info(f"开始计算准确性奖励，样本数: {len(completions)}")
        
        try:
            # 单样本快速路径
            if len(completions) == 1:
                result = process_accuracy(completions[0], solution[0]['content'])
                return [result]
            
            # 多样本并行处理
            pool = self._get_pool()
            tasks = [(comp, sol['content']) for comp, sol in zip(completions, solution)]
            results = pool.starmap(process_accuracy, tasks)
            
            logger.info(f"准确性奖励计算完成: {results}")
            return results
            
        except Exception as e:
            logger.error(f"准确性奖励计算失败: {str(e)}")
            return [0.0] * len(completions)
    
    def __del__(self):
        """清理进程池"""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()


class Table2Latexform(ORM):
    """格式正确性奖励（优化版：支持批量并行）"""
    
    def __init__(self):
        super().__init__()
        self.pool = None
        self._pool_lock = threading.Lock()
    
    def _get_pool(self):
        """懒加载进程池"""
        if self.pool is None:
            with self._pool_lock:
                if self.pool is None:
                    num_workers = min(cpu_count(), 8)
                    self.pool = Pool(processes=num_workers, initializer=init_worker)
                    logger.info(f"创建进程池，worker数量: {num_workers}")
        return self.pool
    
    def __call__(self, completions: List[str], solution: List[dict], **kwargs) -> List[float]:
        """
        批量计算格式奖励
        Args:
            completions: 生成的输出列表
            solution: 标准答案列表
        Returns:
            奖励分数列表
        """
        logger.info(f"开始计算格式奖励，样本数: {len(completions)}")
        
        try:
            # 单样本快速路径
            if len(completions) == 1:
                result = process_format(completions[0], solution[0]['content'])
                return [result]
            
            # 多样本并行处理
            pool = self._get_pool()
            tasks = [(comp, sol['content']) for comp, sol in zip(completions, solution)]
            results = pool.starmap(process_format, tasks)
            
            logger.info(f"格式奖励计算完成: {results}")
            return results
            
        except Exception as e:
            logger.error(f"格式奖励计算失败: {str(e)}")
            return [0.0] * len(completions)
    
    def __del__(self):
        """清理进程池"""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()


# 注册ORM
orms['external_table2latex_acc'] = Table2LatexAcc
orms['external_table2latex_form'] = Table2Latexform

