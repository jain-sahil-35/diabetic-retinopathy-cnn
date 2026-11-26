import argparse, copy, json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, cohen_kappa_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = dict(filters=(32,64,128), hidden=(128,), activation='ReLU', dropout=.3)
CHANGES = [{}, {'filters':(16,32,64)}, {'filters':(64,128,256)}, {'filters':(32,64)}, {'filters':(32,64,128,256)}, {'filters':(32,64,128,256,256)}, {'hidden':()}, {'hidden':(128,64)}, {'hidden':(128,64,32)}, {'hidden':(64,)}, {'hidden':(256,)}, {'hidden':(512,)}, {'activation':'LeakyReLU'}, {'activation':'ELU'}, {'activation':'Tanh'}, {'dropout':0.}, {'dropout':.2}, {'dropout':.5}]
CONFIGS = {f'M{i:02d}': BASE | c for i,c in enumerate(CHANGES,1)}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class DRDataset(Dataset):
    def __init__(self, frame, folder, size, training=False):
        self.frame = frame.reset_index(drop=True)
        self.paths = {p.stem:p for p in Path(folder).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'}}
        missing = set(self.frame.image)-self.paths.keys()
        if missing: raise ValueError(f'Missing images: {list(missing)[:5]}')
        self.transform = T.Compose([T.Resize((size,size))] + ([T.RandomHorizontalFlip(), T.RandomRotation(10)] if training else []) + [T.ToTensor(), T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        row = self.frame.iloc[i]
        with Image.open(self.paths[row.image]) as im: x = self.transform(im.convert('RGB'))
        return x, int(row.grade)

def read_labels(path):
    df = pd.read_csv(path)
    df = df.iloc[:,:2].copy(); df.columns=['image','grade']
    df['image'] = df.image.astype(str).map(lambda x: Path(x.strip()).stem)
    df['grade'] = pd.to_numeric(df.grade, errors='raise')
    if not df.grade.isin(range(5)).all() or df.image.duplicated().any(): raise ValueError('Invalid grades or duplicate image IDs')
    return df

class CNN(nn.Module):
    def __init__(self, filters, hidden, activation, dropout):
        super().__init__()
        act = getattr(nn, activation)
        layers=[]; channels=3
        for f in filters:
            layers += [nn.Conv2d(channels,f,3,padding=1), act(), nn.MaxPool2d(2)]
            channels=f
        self.features=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten())
        layers=[]
        for n in hidden:
            layers += [nn.Linear(channels,n),act(),nn.Dropout(dropout)]; channels=n
        self.classifier=nn.Sequential(*layers,nn.Linear(channels,5))
    def forward(self,x): return self.classifier(self.features(x))

def metrics(y,p):
    return dict(accuracy=accuracy_score(y,p), precision=precision_score(y,p,labels=range(5),average='macro',zero_division=0), recall=recall_score(y,p,labels=range(5),average='macro',zero_division=0), f1=f1_score(y,p,labels=range(5),average='macro',zero_division=0), weighted_f1=f1_score(y,p,average='weighted',zero_division=0), kappa=cohen_kappa_score(y,p,labels=range(5),weights='quadratic'))

def epoch(model,loader,loss_fn,device,optimizer=None):
    model.train(optimizer is not None); total=0.; y=[]; p=[]
    with torch.set_grad_enabled(optimizer is not None):
        for x,t in loader:
            x,t=x.to(device),t.to(device)
            if optimizer is not None: optimizer.zero_grad()
            logits=model(x); loss=loss_fn(logits,t)
            if optimizer is not None: loss.backward(); optimizer.step()
            total += loss.item()*len(t); y.extend(t.cpu().tolist()); p.extend(logits.argmax(1).cpu().tolist())
    return total/len(y), metrics(y,p), y,p

def save_eval(folder,y,p):
    pd.DataFrame(classification_report(y,p,labels=range(5),output_dict=True,zero_division=0)).T.to_csv(folder/'classification_report.csv')
    cm=confusion_matrix(y,p,labels=range(5)); pd.DataFrame(cm).to_csv(folder/'confusion_matrix.csv',index=False)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues'); plt.tight_layout(); plt.savefig(folder/'confusion_matrix.png'); plt.close()

def main():
    ap=argparse.ArgumentParser()
    for name in ['train-images','train-csv','test-images','test-csv']: ap.add_argument('--'+name,required=True)
    ap.add_argument('--out',default='results'); ap.add_argument('--epochs',type=int,default=50); ap.add_argument('--patience',type=int,default=10); ap.add_argument('--batch',type=int,default=16); ap.add_argument('--size',type=int,default=512); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--models',nargs='+',choices=list(CONFIGS),default=list(CONFIGS)); ap.add_argument('--final-test',action='store_true'); a=ap.parse_args()
    if a.epochs<1 or a.patience<1 or a.batch<1 or a.size<32: ap.error('Invalid training settings')
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    df=read_labels(a.train_csv); train,val=train_test_split(df,test_size=.2,random_state=a.seed,stratify=df.grade)
    train.to_csv(out/'train_split.csv',index=False); val.to_csv(out/'validation_split.csv',index=False)
    train_ds=DRDataset(train,a.train_images,a.size,True); val_ds=DRDataset(val,a.train_images,a.size)
    counts=np.bincount(train.grade.astype(int),minlength=5)
    if (counts==0).any(): raise ValueError('Training split lacks a class')
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(len(train)/(5*counts),dtype=torch.float32,device=device))
    rows=[]
    for name in a.models:
        seed_all(a.seed); cfg=CONFIGS[name]; folder=out/name; folder.mkdir(exist_ok=True)
        tr=DataLoader(train_ds,batch_size=a.batch,shuffle=True,generator=torch.Generator().manual_seed(a.seed)); va=DataLoader(val_ds,batch_size=a.batch)
        model=CNN(**cfg).to(device); opt=torch.optim.Adam(model.parameters(),lr=.001)
        best=-1.; stale=0; history=[]; start=time.perf_counter()
        for e in range(1,a.epochs+1):
            tl,tm,_,_=epoch(model,tr,loss_fn,device,opt); vl,vm,_,_=epoch(model,va,loss_fn,device)
            history.append(dict(epoch=e,train_loss=tl,val_loss=vl,train_accuracy=tm['accuracy'],val_accuracy=vm['accuracy'],val_f1=vm['f1']))
            print(name,e,'val F1',round(vm['f1'],4),flush=True)
            if vm['f1']>best:
                best=vm['f1']; stale=0; best_epoch=e; torch.save(model.state_dict(),folder/'best.pt')
            else: stale+=1
            if stale>=a.patience: break
        model.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True)); vl,vm,y,p=epoch(model,va,loss_fn,device)
        rows.append(dict(model=name,**cfg,conv_layers=len(cfg['filters']),dense_layers=len(cfg['hidden']),parameters=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,seconds=time.perf_counter()-start,val_loss=vl,**vm))
        pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False); save_eval(folder,y,p)
        h=pd.DataFrame(history); h.to_csv(folder/'history.csv',index=False)
        for cols,filename in [(['train_accuracy','val_accuracy'],'accuracy'),(['train_loss','val_loss'],'loss')]:
            h.plot(x='epoch',y=cols); plt.tight_layout(); plt.savefig(folder/f'{filename}.png'); plt.close()
    results=pd.DataFrame(rows); results.set_index('model')[['accuracy','precision','recall','f1']].plot.bar(figsize=(14,6)); plt.ylim(0,1); plt.ylabel('Score'); plt.tight_layout(); plt.savefig(out/'comparison.png'); plt.close()
    winner=max(rows,key=lambda r:r['f1'])['model']; print('Best validation model:',winner)
    if a.final_test:
        test=read_labels(a.test_csv); ds=DRDataset(test,a.test_images,a.size); model=CNN(**CONFIGS[winner]).to(device); model.load_state_dict(torch.load(out/winner/'best.pt',map_location=device,weights_only=True)); _,m,y,p=epoch(model,DataLoader(ds,batch_size=a.batch),loss_fn,device); folder=out/'final_test'; folder.mkdir(exist_ok=True); save_eval(folder,y,p); (folder/'metrics.json').write_text(json.dumps({'model':winner,**m},indent=2)); print(m)

if __name__=='__main__': main()
