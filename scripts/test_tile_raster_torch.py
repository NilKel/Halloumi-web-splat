#!/usr/bin/env python3
"""
PyTorch tile rasterizer test — renders one camera view fast.

Usage:
    conda run -n nest_splatting python scripts/test_tile_raster_torch.py \
        --model_path .../betscaled --baked_dir .../baked_atlas \
        --cameras .../cameras.json --cam_idx 0
"""
import os, sys, json, pickle, math, argparse, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/home/nilkel/Projects/nest-splatting")

SH_C0=0.28209479177387814; SH_C1=0.4886025119029199
SH_C2=[1.0925484305920792,-1.0925484305920792,0.31539156525252005,-1.0925484305920792,0.5462742152960396]
SH_C3=[-0.5900435899266435,2.890611442640554,-0.4570457994644658,0.3731763325901154,-0.4570457994644658,1.445305721320277,-0.5900435899266435]

def eval_sh(dirs, sh, deg=3):
    x,y,z=dirs[:,0],dirs[:,1],dirs[:,2]
    r=SH_C0*sh[:,0]
    if deg>0: r=r+(-SH_C1*y[:,None]*sh[:,1]+SH_C1*z[:,None]*sh[:,2]-SH_C1*x[:,None]*sh[:,3])
    if deg>1:
        xx,yy,zz=x*x,y*y,z*z; xy,yz,xz=x*y,y*z,x*z
        r=r+(SH_C2[0]*xy[:,None]*sh[:,4]+SH_C2[1]*yz[:,None]*sh[:,5]+SH_C2[2]*(2*zz-xx-yy)[:,None]*sh[:,6]+SH_C2[3]*xz[:,None]*sh[:,7]+SH_C2[4]*(xx-yy)[:,None]*sh[:,8])
    if deg>2:
        r=r+(SH_C3[0]*(y*(3*xx-yy))[:,None]*sh[:,9]+SH_C3[1]*(xy*z)[:,None]*sh[:,10]+SH_C3[2]*(y*(4*zz-xx-yy))[:,None]*sh[:,11]+SH_C3[3]*(z*(2*zz-3*xx-3*yy))[:,None]*sh[:,12]+SH_C3[4]*(x*(4*zz-xx-yy))[:,None]*sh[:,13]+SH_C3[5]*(z*(xx-yy))[:,None]*sh[:,14]+SH_C3[6]*(x*(xx-3*yy))[:,None]*sh[:,15])
    return (r+0.5).clamp(min=0)

def quat_to_rotmat(q):
    w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3]; N=len(q)
    R=torch.zeros(N,3,3,device=q.device)
    R[:,0,0]=1-2*(y*y+z*z);R[:,0,1]=2*(x*y-w*z);R[:,0,2]=2*(x*z+w*y)
    R[:,1,0]=2*(x*y+w*z);R[:,1,1]=1-2*(x*x+z*z);R[:,1,2]=2*(y*z-w*x)
    R[:,2,0]=2*(x*z-w*y);R[:,2,1]=2*(y*z+w*x);R[:,2,2]=1-2*(x*x+y*y)
    return R

def load_camera(cam, W=800, H=800):
    R=np.array(cam["rotation"],dtype=np.float32); t=np.array(cam["position"],dtype=np.float32)
    v=np.eye(4,dtype=np.float32); v[:3,:3]=R; v[:3,3]=-R@t
    zn,zf=0.01,100.0; fx,fy=cam["fx"],cam["fy"]
    tx,ty_=W/(2*fx),H/(2*fy); r,tp=tx*zn,ty_*zn
    p=np.zeros((4,4),dtype=np.float32)
    p[0,0]=2*zn/(2*r);p[1,1]=2*zn/(2*tp);p[2,2]=zf/(zf-zn);p[2,3]=-(zf*zn)/(zf-zn);p[3,2]=1
    return v,p,t.copy(),fx,fy

def render_cuda_ref(model_path, baked_dir, cam, W, H):
    from scene import GaussianModel; from hash_encoder.config import Config
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer
    with open(os.path.join(model_path,"args.pkl"),'rb') as f: args=pickle.load(f)
    cy=os.path.join(model_path,"config.yaml")
    cfg=Config(cy) if os.path.exists(cy) else Config(args.yaml)
    g=GaussianModel(sh_degree=3); g.load_ply(os.path.join(baked_dir,"baked.ply"))
    g.active_sh_degree=3; g.base_opacity=cfg.surfel.tg_base_alpha
    kn=getattr(args,'kernel','gaussian'); km={'gaussian':0,'beta':1,'flex':2,'general':3,'beta_scaled':4}
    kt=km.get(kn,0)
    v,p,cp,fx,fy=load_camera(cam,W,H); fp=p@v
    vt=torch.tensor(v.T,dtype=torch.float32,device="cuda")
    fpt=torch.tensor(fp.T,dtype=torch.float32,device="cuda")
    cpt=torch.tensor(cp,dtype=torch.float32,device="cuda")
    rs=GaussianRasterizationSettings(image_height=H,image_width=W,tanfovx=W/(2*fx),tanfovy=H/(2*fy),
        bg=torch.zeros(3,dtype=torch.float32,device="cuda"),scale_modifier=1.0,
        viewmatrix=vt,projmatrix=fpt,sh_degree=3,campos=cpt,prefiltered=False,debug=False,
        beta=cfg.surfel.tg_beta,aabb_mode=3)
    r=GaussianRasterizer(raster_settings=rs)
    sh=g.get_shape if hasattr(g,'_shape') and g._shape is not None else None
    with torch.no_grad():
        c,rad=r(means3D=g.get_xyz,means2D=torch.zeros_like(g.get_xyz[:,:2]),
            opacities=g.get_opacity,shs=g.get_features,scales=g.get_scaling,
            rotations=g.get_rotation,shapes=sh,kernel_type=kt,
            residual_textures=None,atlas_texture=None,atlas_rects=None,atlas_width=0)
    return c.clamp(0,1).permute(1,2,0).cpu().numpy(), g, args, cfg

@torch.no_grad()
def preprocess(g, view, proj, campos, W, H, kt):
    dev='cuda'
    m3d=g.get_xyz; sc=g.get_scaling; rot=g.get_rotation
    opa=g.get_opacity.squeeze(); shs=g.get_features
    sh_t=g.get_shape if hasattr(g,'_shape') and g._shape is not None else None
    shapes=sh_t.squeeze() if sh_t is not None else None
    N=len(m3d)
    fp=torch.tensor((proj@view).T,dtype=torch.float32,device=dev)  # CUDA uses PV^T via GLM reindexing
    n2p=torch.tensor([[W/2,0,0],[0,H/2,0],[0,0,0],[(W-1)/2,(H-1)/2,1]],dtype=torch.float32,device=dev)
    vt=torch.tensor(view,dtype=torch.float32,device=dev)
    ct=torch.tensor(campos,dtype=torch.float32,device=dev)
    pv=(vt[:3,:3]@m3d.T).T+vt[:3,3]
    dm=pv[:,2]>0.01
    R=quat_to_rotmat(rot); L=torch.zeros(N,3,3,device=dev)
    L[:,:,0]=R[:,:,0]*sc[:,0:1]; L[:,:,1]=R[:,:,1]*sc[:,1:2]; L[:,:,2]=R[:,:,2]
    s2w=torch.zeros(N,4,3,device=dev)
    s2w[:,:3,0]=L[:,:,0]; s2w[:,:3,1]=L[:,:,1]; s2w[:,:3,2]=m3d; s2w[:,3,2]=1
    fpn=fp@n2p; Tm=s2w.permute(0,2,1)@fpn
    Tu,Tv,Tw=Tm[:,:,0],Tm[:,:,1],Tm[:,:,2]
    # AdR cutoff
    co=torch.full((N,),4.0,device=dev)
    if kt in[1,4] and shapes is not None:
        ksq=9.0 if kt==4 else 1.0; k=3.0 if kt==4 else 1.0
        rat=1.0/(255*opa.clamp(min=1e-10)); thr=rat.pow(1.0/shapes.clamp(min=1e-10))
        rb=torch.where(thr<1,k*(1-thr).clamp(min=0).sqrt(),torch.zeros_like(thr))
        lt=(255*opa.clamp(min=1e-10)).log(); rl=torch.where(lt>0,(2*lt).sqrt(),torch.zeros_like(lt))
        co=torch.maximum(rb,rl).clamp(max=k+2)
    om=opa>=1/255
    csq=co**2; Tsq=Tw*Tw
    dv=csq*Tsq[:,0]+csq*Tsq[:,1]-Tsq[:,2]; vd=dv.abs()>1e-10
    f0=torch.where(vd,csq/dv,torch.zeros_like(dv)); f2=torch.where(vd,-1/dv,torch.zeros_like(dv))
    cx=f0*Tu[:,0]*Tw[:,0]+f0*Tu[:,1]*Tw[:,1]+f2*Tu[:,2]*Tw[:,2]
    cy_=f0*Tv[:,0]*Tw[:,0]+f0*Tv[:,1]*Tw[:,1]+f2*Tv[:,2]*Tw[:,2]
    hxs=cx*cx-(f0*Tu[:,0]**2+f0*Tu[:,1]**2+f2*Tu[:,2]**2)
    hys=cy_*cy_-(f0*Tv[:,0]**2+f0*Tv[:,1]**2+f2*Tv[:,2]**2)
    hx=hxs.clamp(min=1e-4).sqrt(); hy=hys.clamp(min=1e-4).sqrt()
    vm=dm&om&vd&(hx<W*2)&(hy<H*2)&(cx>-500)&(cx<W+500)&(cy_>-500)&(cy_<H+500)
    vi=torch.where(vm)[0]; V=len(vi)
    print(f"[PRE] {V}/{N} visible")
    d=m3d[vi]-ct; d=d/(d.norm(dim=1,keepdim=True)+1e-8)
    col=eval_sh(d,shs[vi])
    return {'vi':vi,'Tu':Tu[vi],'Tv':Tv[vi],'Tw':Tw[vi],'cx':cx[vi],'cy':cy_[vi],
            'hx':hx[vi],'hy':hy[vi],'col':col,'opa':opa[vi],'dep':pv[vi,2],
            'shp':shapes[vi] if shapes is not None else None}

def tile_bin(d, W, H):
    """Fully vectorized tile binning using torch on GPU."""
    TS=16; tx_=(W+TS-1)//TS; ty_=(H+TS-1)//TS; tt=tx_*ty_
    V=len(d['vi']); dev='cuda'
    cx,cy=d['cx'],d['cy']; hx,hy=d['hx'],d['hy']; dep=d['dep']
    rx=torch.maximum(torch.ceil(hx),torch.ones(1,device=dev)).int()
    ry=torch.maximum(torch.ceil(hy),torch.ones(1,device=dev)).int()
    rmx=torch.clamp(((cx-rx.float())/TS).int(),min=0)
    rmy=torch.clamp(((cy-ry.float())/TS).int(),min=0)
    rMx=torch.clamp(torch.ceil((cx+rx.float())/TS).int(),max=tx_)
    rMy=torch.clamp(torch.ceil((cy+ry.float())/TS).int(),max=ty_)
    rMx=torch.maximum(rMx,rmx); rMy=torch.maximum(rMy,rmy)
    ntx=(rMx-rmx); nty=(rMy-rmy); nt=ntx*nty
    te=int(nt.sum().item())
    print(f"[BIN] {tx_}x{ty_}={tt} tiles, {V} vis, {te} entries ({te/max(V,1):.1f}/g)")
    # Vectorized: expand each Gaussian into its tiles
    # For each Gaussian, generate all (ty_off, tx_off) pairs
    vis_ids_g=torch.arange(V,device=dev).repeat_interleave(nt)
    # We need tile coords. Use cumsum trick: for each Gaussian with nt[i] tiles,
    # generate offsets 0..nt[i]-1 and map to (ty, tx) within its rect
    local_idx=torch.arange(te,device=dev)
    offsets=torch.zeros(V+1,dtype=torch.long,device=dev); offsets[1:]=torch.cumsum(nt.long(),0)
    local_off=local_idx-offsets[:-1].repeat_interleave(nt)
    # Map local offset to (ty_off, tx_off) within rect
    ntx_rep=ntx.repeat_interleave(nt)
    ty_off=local_off//ntx_rep; tx_off=local_off%ntx_rep
    rmy_rep=rmy.repeat_interleave(nt); rmx_rep=rmx.repeat_interleave(nt)
    tile_ty=rmy_rep+ty_off; tile_tx=rmx_rep+tx_off
    tile_ids_g=(tile_ty*tx_+tile_tx).int()
    dep_rep=dep.repeat_interleave(nt)
    # Sort by (tile_id, depth)
    sk=tile_ids_g.long()*1000000+(dep_rep*10000).long()
    so=torch.argsort(sk)
    tile_ids_s=tile_ids_g[so]; vis_ids_s=vis_ids_g[so]
    # Build tile ranges
    tr=torch.zeros(tt,2,dtype=torch.int32,device=dev)
    if te>0:
        diff=torch.diff(tile_ids_s)
        ch=torch.where(diff!=0)[0]+1
        ss=torch.cat([torch.zeros(1,dtype=torch.long,device=dev),ch])
        es=torch.cat([ch,torch.tensor([te],dtype=torch.long,device=dev)])
        tr[tile_ids_s[ss],0]=ss.int(); tr[tile_ids_s[ss],1]=es.int()
    ne=int((tr[:,1]>tr[:,0]).sum().item())
    mp=int((tr[:,1]-tr[:,0]).max().item()) if te>0 else 0
    print(f"[BIN] Non-empty: {ne}, max/tile: {mp}")
    return tr.cpu().numpy(), vis_ids_s.cpu().numpy(), tx_, ty_

@torch.no_grad()
def tile_raster(tr, sv, d, W, H, kt, tx_, ty_):
    """Vectorized: all pixels in tile processed in parallel per Gaussian."""
    dev='cuda'; TS=16
    img=torch.zeros(H,W,3,device=dev)
    T_map=torch.ones(H,W,device=dev)
    Tu,Tv,Tw=d['Tu'],d['Tv'],d['Tw']
    col,opa=d['col'],d['opa']
    shp=d['shp']; cxd,cyd=d['cx'],d['cy']
    FIS=1/(2*0.3*0.3); has_shp=shp is not None
    sv_t=torch.tensor(sv,dtype=torch.long,device=dev)
    tt=tx_*ty_; ne=int(np.sum(tr[:,1]>tr[:,0]))
    t0=time.time(); done_cnt=0
    for tid in range(tt):
        s,e=tr[tid]
        if s==e: continue
        ty=tid//tx_; tx=tid%tx_
        y0=ty*TS; y1=min(y0+TS,H); x0=tx*TS; x1=min(x0+TS,W)
        ph,pw=y1-y0,x1-x0
        py=torch.arange(y0,y1,device=dev,dtype=torch.float32)
        px=torch.arange(x0,x1,device=dev,dtype=torch.float32)
        gy,gx=torch.meshgrid(py,px,indexing='ij')
        gx=gx.reshape(-1); gy=gy.reshape(-1); P=len(gx)
        Tp=torch.ones(P,device=dev); cp=torch.zeros(P,3,device=dev)
        dn=torch.zeros(P,dtype=torch.bool,device=dev)
        tvis=sv_t[s:e]; ng=len(tvis)
        for j in range(ng):
            if dn.all(): break
            vi=tvis[j]; tu,tv,tw=Tu[vi],Tv[vi],Tw[vi]
            kx=gx*tw[0]-tu[0]; ky=gx*tw[1]-tu[1]; kz=gx*tw[2]-tu[2]
            lx=gy*tw[0]-tv[0]; ly=gy*tw[1]-tv[1]; lz=gy*tw[2]-tv[2]
            pcx=ky*lz-kz*ly; pcy=kz*lx-kx*lz; pcz=kx*ly-ky*lx
            ok=pcz.abs()>1e-10
            ipz=torch.where(ok,1.0/pcz,torch.zeros_like(pcz))
            sx=pcx*ipz; sy=pcy*ipz; r3=sx*sx+sy*sy
            dx=cxd[vi]-gx; dy=cyd[vi]-gy; r2=FIS*(dx*dx+dy*dy)
            dep=torch.where(r3<=r2,sx*tw[0]+sy*tw[1]+tw[2],tw[2].expand(P))
            ok=ok&(dep>=0.01)
            o=opa[vi]
            if kt==4 and has_shp:
                ok=ok&(r3<9+1e-6)
                b=(1-r3/9).clamp(min=0); ab=b.pow(shp[vi])
                al=(-r2/2).exp(); kv=torch.maximum(ab,al)
                a=(o*kv).clamp(max=0.99)
            else:
                rho=torch.minimum(r3,r2); a=(o*(-0.5*rho).exp()).clamp(max=0.99)
            ok=ok&(a>=1/255)&(~dn)
            w=a*Tp; cp+=ok.unsqueeze(1)*w.unsqueeze(1)*col[vi]
            Tp=torch.where(ok,Tp*(1-a),Tp)
            dn=dn|(ok&(Tp<0.0001))
        img[y0:y1,x0:x1]=cp.reshape(ph,pw,3)
        T_map[y0:y1,x0:x1]=Tp.reshape(ph,pw)
        done_cnt+=1
        if done_cnt%500==0:
            print(f"  {done_cnt}/{ne} tiles, {time.time()-t0:.1f}s")
    print(f"  Done: {done_cnt} tiles in {time.time()-t0:.1f}s")
    return img.cpu().numpy(), T_map.cpu().numpy()

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument("--model_path",required=True); pa.add_argument("--baked_dir",required=True)
    pa.add_argument("--cameras",required=True); pa.add_argument("--cam_idx",type=int,default=0)
    pa.add_argument("--out_dir",default="/tmp/tile_raster_test")
    a=pa.parse_args()
    os.makedirs(a.out_dir,exist_ok=True)
    with open(a.cameras) as f: cams=json.load(f)
    cam=cams[a.cam_idx]; W,H=cam["width"],cam["height"]
    print(f"Cam {a.cam_idx}: {cam['img_name']}, {W}x{H}")

    print("\n=== CUDA Ref ===")
    ref,g,ta,cfg=render_cuda_ref(a.model_path,a.baked_dir,cam,W,H)
    Image.fromarray((ref*255).clip(0,255).astype(np.uint8)).save(os.path.join(a.out_dir,"ref_cuda.png"))

    print("\n=== Preprocess ===")
    v,p,cp,*_=load_camera(cam,W,H)
    kn=getattr(ta,'kernel','gaussian')
    kt={'gaussian':0,'beta':1,'flex':2,'general':3,'beta_scaled':4}.get(kn,0)
    d=preprocess(g,v,p,cp,W,H,kt)

    print("\n=== Tile Bin ===")
    tr,sv,txn,tyn=tile_bin(d,W,H)

    print("\n=== Rasterize ===")
    img,T_map=tile_raster(tr,sv,d,W,H,kt,txn,tyn)
    Image.fromarray((img*255).clip(0,255).astype(np.uint8)).save(os.path.join(a.out_dir,"test_tile.png"))

    print("\n=== Compare ===")
    diff=np.abs(ref-img); mse=np.mean(diff**2); psnr=-10*np.log10(mse+1e-10)
    print(f"  PSNR={psnr:.2f}dB MSE={mse:.6f} Max={diff.max():.4f} Mean={diff.mean():.6f}")
    Image.fromarray(((diff*10).clip(0,1)*255).astype(np.uint8)).save(os.path.join(a.out_dir,"diff_10x.png"))
    print(f"  Output: {a.out_dir}/")

if __name__=="__main__": main()
