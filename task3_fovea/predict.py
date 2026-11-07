"""最终融合模型：读取统一目录中的模型，生成中心凹坐标提交。"""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
import model_utils as b
from networks import ConvNextFoveaNet


def main():
    project=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='D:/2026task3/detection')
    p.add_argument('--output',default=str(project/'submission.csv'))
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--invisible-ids',type=int,nargs='*',default=[])
    a=p.parse_args();config=json.loads((project/'model_config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;device=torch.device(a.device)
    cache={};components=[];visibility={};records=None
    if abs(sum(f['weight'] for f in config['families'])-1)>1e-8:raise ValueError('Invalid model weights')
    for family in config['families']:
        side=family['side']
        if side not in cache:cache[side]=b.read_data(a.data,side)
        train,test=cache[side];ids=[r['id'] for r in test]
        if records is None:records=test
        if ids!=[r['id'] for r in records]:raise ValueError('Image order mismatch')
        predictions=[];confidences=[]
        for filename in family['checkpoints']:
            checkpoint=torch.load(project/family['directory']/filename,map_location='cpu',weights_only=True)
            if checkpoint['side']!=side:raise ValueError('Model resolution mismatch')
            if set(ids)&set(checkpoint['train_ids']):raise ValueError('Train/test IDs overlap')
            model=(b.FoveaNet() if family['architecture']=='resnet50_fpn' else ConvNextFoveaNet()).to(device)
            model.load_state_dict(checkpoint['model'])
            xy,vis,_=b.predict_records(model,test,device,a.batch,tta=True)
            predictions.append(xy);confidences.append(vis);del model,checkpoint
        components.append(family['weight']*np.mean(predictions,axis=0))
        visibility[family['name']]=np.mean(confidences,axis=0)
        print(f'{family["name"]}: {len(predictions)} models complete',flush=True)
    points=np.sum(components,axis=0)
    base_visibility=(visibility['resnet_cv']+visibility['convnext_cv'])/2
    if set(a.invisible_ids)-set(ids):raise ValueError('Unknown invisible ID')
    for i,r in enumerate(records):
        points[i]=np.clip(points[i],[0,0],np.array(r['size'])-1)
        if base_visibility[i]<config['visibility_threshold'] or r['id'] in a.invisible_ids:points[i]=0
    if not np.isfinite(points).all():raise ValueError('Non-finite coordinate')
    with (Path(a.data)/'sample_submission.csv').open(encoding='utf-8-sig',newline='') as f:template=list(csv.DictReader(f))
    if len(template)!=2*len(records):raise ValueError('Template coverage mismatch')
    lookup={r['id']:xy for r,xy in zip(records,points)};rows=[];seen=set()
    for row in template:
        stem,axis=row['ImageID'].rsplit('_Fovea_',1);image_id=int(Path(stem).stem)
        if image_id not in lookup or axis not in ('X','Y') or (image_id,axis) in seen:raise ValueError('Invalid template ID')
        seen.add((image_id,axis));rows.append(dict(ImageID=row['ImageID'],value=f'{lookup[image_id][0 if axis=="X" else 1]:.6f}'))
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['ImageID','value']);writer.writeheader();writer.writerows(rows)
    print(f'Verified {len(rows)} coordinate rows: {path}',flush=True)


if __name__=='__main__':main()
