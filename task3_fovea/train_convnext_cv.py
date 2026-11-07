"""Train independent ConvNeXt-FPN models on the same SHA256 grouped folds."""
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import model_utils as b
from networks import ConvNextFoveaNet
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='D:/2026task3/detection')
    p.add_argument('--output',default=str(Path(__file__).resolve().parent/'models'/'convnext_cv'))
    p.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--side',type=int,default=448)
    p.add_argument('--epochs',type=int,default=120)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--eval-every',type=int,default=5)
    p.add_argument('--seed',type=int,default=7641)
    p.add_argument('--ema',type=float,default=.99)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train,test=b.read_data(a.data,a.side)
    splits=list(b.grouped_splits(train,5,2026))
    for fold in a.folds:
        b.seed_everything(a.seed+fold)
        ti,vi=splits[fold]; tr=[train[i] for i in ti]; val=[train[i] for i in vi]
        model=ConvNextFoveaNet(pretrained=True).to(device)
        ema=copy.deepcopy(model).eval()
        for param in ema.parameters(): param.requires_grad_(False)
        decoder={id(v) for m in [model.lateral,model.smooth,model.head,model.presence] for v in m.parameters()}
        opt=torch.optim.AdamW([
            {'params':[v for v in model.parameters() if id(v) not in decoder],'lr':a.lr*.1},
            {'params':[v for v in model.parameters() if id(v) in decoder],'lr':a.lr}],weight_decay=1e-4)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=a.lr*.005)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
        cache=b.tensors(tr,device); best=float('inf'); history=[]; since=time.time()
        for epoch in range(1,a.epochs+1):
            model.train(); model.freeze_batchnorm(); losses=[]
            for ids in torch.randperm(len(tr),device=device).split(a.batch):
                images,points,boxes,visible=b.augment(*[v[ids] for v in cache],negative_rate=.08)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                    loss=b.loss_fn(model(images),points,boxes,visible)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                scaler.step(opt); scaler.update(); losses.append(loss.detach().item())
                with torch.no_grad():
                    for dst,src in zip(ema.parameters(),model.parameters()): dst.lerp_(src,1-a.ema)
                    for dst,src in zip(ema.buffers(),model.buffers()): dst.copy_(src)
            schedule.step()
            if epoch%a.eval_every==0 or epoch==a.epochs:
                result={}
                for kind,net in [('raw',model),('ema',ema)]:
                    xy,vis,boxes=b.predict_records(net,val,device,a.batch)
                    rmse=b.score(val,xy,vis,.1); result[kind]=rmse
                    if rmse<best:
                        best=rmse
                        extra=dict(architecture='convnext_tiny_fpn',fold=fold,side=a.side,epoch=epoch,kind=kind,val_rmse=rmse,args=vars(a),
                            train_ids=[r['id'] for r in tr],val_ids=[r['id'] for r in val],split_strategy='sha256_grouped_kfold_v1',
                            train_sha256=[hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in tr],
                            val_sha256=[hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() for r in val])
                        torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},**extra),out/f'fold_{fold}.pt')
                        np.savez(out/f'fold_{fold}_val.npz',ids=np.array(extra['val_ids']),xy=xy,visibility=vis,boxes=boxes)
                item=dict(epoch=epoch,loss=float(np.mean(losses)),best=best,seconds=round(time.time()-since,1),**result)
                history.append(item); print(json.dumps(dict(fold=fold,**item)),flush=True)
                (out/f'history_{fold}.json').write_text(json.dumps(history,indent=2))
        del model,ema,opt,cache
        if device.type=='cuda': torch.cuda.empty_cache()


if __name__=='__main__': main()
