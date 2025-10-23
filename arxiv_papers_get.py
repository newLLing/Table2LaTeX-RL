import requests
import os
import time
from bs4 import BeautifulSoup
from datetime import datetime


def download_and_extract_source(arxiv_id, id, timeout=10):
    """下载并解压 Arxiv 论文源码，设置超时"""
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 200:
            tar_path = f"D:/{id}_arxiv_papers/{arxiv_id}.tar.gz"  #保存路径
            with open(tar_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Failed to download {arxiv_id}, status code {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"Timeout occurred while downloading {arxiv_id}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return False
    
# 例如爬取2024年10月所有论文
# 创建保存路径
if not os.path.exists("D:/2410_arxiv_papers"):  # 修改保存路径为D盘
    os.makedirs("D:/2410_arxiv_papers")

all_arxiv_ids = []
id = "2410"
#####设置论文数量#####
for i in range(16945,0,-1):
    j = str(i)
    zero = (5-len(j))*"0"
    all_arxiv_ids.append(id + "." + zero + str(i) + "v1")

L = len(all_arxiv_ids)
print(f"Total papers found: {L}")
for arxiv_id in all_arxiv_ids:
    print(f"Processing{arxiv_id}")
    
    download_and_extract_source(arxiv_id, id, timeout=10)
    time.sleep(1)