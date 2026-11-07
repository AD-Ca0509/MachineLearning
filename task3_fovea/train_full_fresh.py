"""Full-data ConvNeXt refits using a fixed CV-derived update budget.

Uses all 80 labels / 79 distinct images; there is no held-out validation score.
"""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_utils as b
from networks import ConvNextFoveaNet
from train_refit import deduplicate
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='D:/2026task3/detection')
    p.add_argument('--output',default=str(Path(__file__).resolve().parent/'models'/'fresh'))
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--schedule-epochs',type=int,default=96)
    p.add_argument('--members',type=int,default=2)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=59017)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train,test=b.read_data(a.data,448);unique=deduplicate(train)
    assert not ({r['id'] for r in train}&{r['id'] for r in test})
    print(f'Full training: {len(train)} labels, {len(unique)} distinct images; fixed {a.epochs} epochs.',flush=True)
    diagnostics=[]
    for member in range(a.members):
        b.seed_everything(a.seed+member)
        model=ConvNextFoveaNet(pretrained=True).to(device);teacher=copy.deepcopy(model).eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
        decoder={id(v) for module in [model.lateral,model.smooth,model.head,model.presence] for v in module.parameters()}
        opt=torch.optim.AdamW([
            {'params':[v for v in model.parameters() if id(v) not in decoder],'lr':5e-5},
            {'params':[v for v in model.parameters() if id(v) in decoder],'lr':5e-4}],weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.schedule_epochs,eta_min=2.5e-6)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        cache=b.tensors(unique,device);history=[];since=time.time()
        for epoch in range(1,a.epochs+1):
            model.train();losses=[]
            for ids in torch.randperm(len(unique),device=device).split(a.batch):
                images,points,boxes,visible=b.augment(*[v[ids] for v in cache],negative_rate=.08)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    loss=b.loss_fn(model(images),points,boxes,visible)
                scaler.scale(loss).backward();scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                scaler.step(opt);scaler.update();losses.append(loss.detach().item())
                with torch.no_grad():
                    for dst,src in zip(teacher.parameters(),model.parameters()):dst.lerp_(src,.01)
                    for dst,src in zip(teacher.buffers(),model.buffers()):dst.copy_(src)
            schedule.step();item=dict(member=member,epoch=epoch,loss=float(np.mean(losses)),seconds=round(time.time()-since,1))
            history.append(item)
            if epoch%5==0:print(json.dumps(item),flush=True)
        # Fixed endpoint averaging, without selecting using test or training-fit scores.
        state={}
        for key,raw in model.state_dict().items():
            ema=teacher.state_dict()[key]
            state[key]=((raw+ema)*.5 if raw.is_floating_point() else raw).detach().cpu()
        model.load_state_dict(state)
        fit_xy,_,_=b.predict_records(model,unique,device,a.batch)
        fit_rmse=float(np.sqrt(np.mean((fit_xy-np.array([r['gt'] for r in unique]))**2)))
        xy,vis,boxes=b.predict_records(model,test,device,a.batch)
        extra=dict(architecture='convnext_tiny_fpn',side=448,member=member,train_ids=[r['id'] for r in unique],
            source_label_ids=[r['id'] for r in train],merged_ids=[r['source_ids'] for r in unique if len(r['source_ids'])>1],
            train_sha256=[hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in unique],
            split_strategy='full_refit_from_sha256_grouped_cv',validation_score=None,training_fit_rmse=fit_rmse,
            args=vars(a),visibility_policy='v2',fixed_endpoint_average='50% raw + 50% EMA')
        torch.save(dict(model=state,**extra),out/f'member_{member}.pt')
        np.savez(out/f'member_{member}_test.npz',ids=np.array([r['id'] for r in test]),xy=xy,visibility=vis,boxes=boxes)
        diagnostics.append(dict(member=member,training_fit_rmse=fit_rmse))
        (out/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        (out/f'history_{member}.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(diagnostics[-1]),flush=True)
        del model,teacher,opt,cache
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':main()
