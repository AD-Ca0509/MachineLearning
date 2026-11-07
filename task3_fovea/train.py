"""最终模型训练入口：按顺序训练各融合成员。"""
import argparse,subprocess,sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='D:/2026task3/detection')
    p.add_argument('--stage',choices=['all','resnet','convnext','refit','fresh'],default='all')
    a=p.parse_args()
    stages=[('resnet','train_resnet_cv.py',['--mode','train']),('convnext','train_convnext_cv.py',[]),('refit','train_refit.py',[]),('fresh','train_full_fresh.py',[])]
    for name,filename,extra in stages:
        if a.stage not in ('all',name):continue
        subprocess.run([sys.executable,'-B',str(root/filename),'--data',a.data,*extra],cwd=root,check=True)

if __name__=='__main__':main()
