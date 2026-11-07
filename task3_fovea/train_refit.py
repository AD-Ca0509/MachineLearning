"""Refit existing CV-selected ConvNeXt models on all training labels.

These production members have no held-out validation score. Their base recipe
was selected by grouped CV. Training-fit diagnostics are not leaderboard scores.
"""
import argparse,copy,hashlib,json,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_utils as b
from networks import ConvNextFoveaNet
import numpy as np
import torch


def deduplicate(records):
    groups={}
    for r in records:
        digest=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()
        groups.setdefault(digest,[]).append(r)
    unique=[]
    for group in groups.values():
        r=group[0].copy();r['source_ids']=[x['id'] for x in group]
        for key in ['point','box','gt']:
            r[key]=np.mean([x[key] for x in group],axis=0).tolist()
        unique.append(r)
    return unique


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='D:/2026task3/detection')
    p.add_argument('--source-models',default=str(Path(__file__).resolve().parent/'models'/'convnext_cv'))
    p.add_argument('--output',default=str(Path(__file__).resolve().parent/'models'/'refit'))
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--backbone-lr',type=float,default=1e-5)
    p.add_argument('--head-lr',type=float,default=5e-5)
    p.add_argument('--ema',type=float,default=.95)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=42017)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train,test=b.read_data(a.data,448);unique=deduplicate(train)
    print(f'Refit: {len(train)} labels, {len(unique)} unique training images, {len(test)} test images; no test labels used.',flush=True)
    diagnostics=[]
    for fold in range(5):
        b.seed_everything(a.seed+fold)
        source=Path(a.source_models)/f'fold_{fold}.pt'
        checkpoint=torch.load(source,map_location='cpu',weights_only=True)
        assert checkpoint['side']==448 and checkpoint['architecture']=='convnext_tiny_fpn'
        model=ConvNextFoveaNet().to(device);model.load_state_dict(checkpoint['model'])
        old_test_xy,old_test_vis,_=b.predict_records(model,test,device,a.batch)
        teacher=copy.deepcopy(model).eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
        decoder={id(v) for module in [model.lateral,model.smooth,model.head,model.presence] for v in module.parameters()}
        optimizer=torch.optim.AdamW([
            {'params':[v for v in model.parameters() if id(v) not in decoder],'lr':a.backbone_lr},
            {'params':[v for v in model.parameters() if id(v) in decoder],'lr':a.head_lr}],weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs,eta_min=1e-7)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        cache=b.tensors(unique,device);history=[];since=time.time()
        for epoch in range(1,a.epochs+1):
            model.train();model.freeze_batchnorm();losses=[]
            for ids in torch.randperm(len(unique),device=device).split(a.batch):
                images,points,boxes,visible=b.augment(*[v[ids] for v in cache],negative_rate=.06)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    loss=b.loss_fn(model(images),points,boxes,visible)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                scaler.step(optimizer);scaler.update();losses.append(loss.detach().item())
                with torch.no_grad():
                    for dst,src in zip(teacher.parameters(),model.parameters()):dst.lerp_(src,1-a.ema)
                    for dst,src in zip(teacher.buffers(),model.buffers()):dst.copy_(src)
            schedule.step();item=dict(source_fold=fold,epoch=epoch,loss=float(np.mean(losses)),seconds=round(time.time()-since,1))
            history.append(item)
            if epoch%3==0:print(json.dumps(item),flush=True)
        fit_xy,fit_vis,_=b.predict_records(teacher,unique,device,a.batch)
        fit_rmse=float(np.sqrt(np.mean((fit_xy-np.array([r['gt'] for r in unique]))**2)))
        xy,vis,boxes=b.predict_records(teacher,test,device,a.batch)
        new=dict(architecture='convnext_tiny_fpn',side=448,source_fold=fold,source_cv_epoch=checkpoint['epoch'],
                 source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 train_ids=[r['id'] for r in unique],source_label_ids=[r['id'] for r in train],
                 merged_ids=[r['source_ids'] for r in unique if len(r['source_ids'])>1],
                 train_sha256=[hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in unique],
                 split_strategy='full_refit_from_sha256_grouped_cv',validation_score=None,
                 training_fit_rmse=fit_rmse,args=vars(a),visibility_policy='v2')
        torch.save(dict(model={k:v.detach().cpu() for k,v in teacher.state_dict().items()},**new),out/f'fold_{fold}.pt')
        np.savez(out/f'fold_{fold}_test.npz',ids=np.array([r['id'] for r in test]),xy=xy,visibility=vis,boxes=boxes,old_xy=old_test_xy,old_visibility=old_test_vis)
        item=dict(source_fold=fold,training_fit_rmse=fit_rmse,test_coordinate_change_rmse=float(np.sqrt(np.mean((xy-old_test_xy)**2))))
        diagnostics.append(item);print(json.dumps(item),flush=True)
        (out/f'history_{fold}.json').write_text(json.dumps(history,indent=2))
        (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        del model,teacher,optimizer,cache
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':main()
