from types import SimpleNamespace as NS
import pytest
import torch
from _patches import vision_preprocessors as p


def test_feature_exit_preserves_capture_and_does_not_mutate_hooks():
    captured=[];tail=[]
    def capture(h,skip,opts):
        captured.append(h.clone());return h,skip
    hooks=[capture]
    kw={'input':torch.ones(1,4,2,2),'timestep':torch.ones(1),
        'c':{'transformer_options':{'patches':{'output_block_patch':hooks}}}}
    def model(x,t,transformer_options):
        h=torch.arange(640.).reshape(1,640,1,1)
        for hook in transformer_options['patches']['output_block_patch']:
            h,_=hook(h,None,{'block':('output',9)})
        tail.append(True)
        return x*2
    result=p._feature_only(model,kw)
    assert torch.equal(captured[0],torch.arange(640.).reshape(1,640,1,1))
    assert not tail and torch.count_nonzero(result)==0
    assert hooks==[capture]


def test_other_model_errors_are_not_swallowed():
    def model(*args,**kwargs):raise ValueError('real failure')
    kw={'input':torch.zeros(1),'timestep':torch.ones(1),'c':{'transformer_options':{'patches':{'output_block_patch':[lambda *a:a]}}}}
    with pytest.raises(ValueError,match='real failure'):p._feature_only(model,kw)


def test_missing_capture_hook_keeps_normal_forward():
    x=torch.ones(1)
    assert torch.equal(p._feature_only(lambda x,t,**kw:x+1,{'input':x,'timestep':x,'c':{}}),x+1)


@pytest.mark.parametrize('device,blocks,custom', [('cpu',12,False),('mps',11,False),('mps',12,True)])
def test_pose_fallback_for_unvalidated_models(device,blocks,custom):
    model=NS(load_device=torch.device(device),model=NS(diffusion_model=NS(heatmap_head=object(),output_blocks=[None]*blocks)),model_options={'model_function_wrapper':object()} if custom else {})
    seen=[]
    def original(*args):seen.append(args[0]);return 'stock'
    assert p._pose_execute(original,model,None,None,16)=='stock'
    assert seen==[model]


@pytest.mark.parametrize('fail',[False,True])
def test_camera_cast_flags_restored(fail):
    heads=[NS(comfy_cast_weights=False),NS(comfy_cast_weights=True),NS(comfy_cast_weights=False)]
    camera=NS(fc_t=heads[0],fc_qvec=heads[1],fc_fov=[heads[2]])
    def original(self,feat):
        assert all(h.comfy_cast_weights for h in heads)
        if fail:raise ValueError('failed camera')
        return 7
    if fail:
        with pytest.raises(ValueError):p._camera_forward(original,camera,NS(device=torch.device('mps')))
    else:assert p._camera_forward(original,camera,NS(device=torch.device('mps')))==7
    assert [h.comfy_cast_weights for h in heads]==[False,True,False]


def test_batch_limits_and_opt_out(monkeypatch):
    for value,want in [('0',1),('1',1),('99',2),('bad',1)]:
        monkeypatch.setenv('ASFP8_DA3_BATCH',value);assert p._batch_size()==want


def test_source_guard_fails_closed(tmp_path):
    f=tmp_path/'source.py';f.write_text('changed')
    assert not p._matches(NS(__file__=str(f)),p._NODE_HASH)
    assert not p._matches(NS(),p._NODE_HASH)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='requires MPS')
def test_da3_odd_frame_count_bounded_and_order_preserved(monkeypatch):
    monkeypatch.setenv('ASFP8_DA3_BATCH','2')
    calls=[];interrupts=[]
    class Diff:
        dtype=torch.float16;head_type='dpt';cam_dec=None
        def __call__(self,x):
            calls.append(len(x));return {'depth':x[:,0],'sky':x[:,1]}
    mm=NS(get_torch_device=lambda:torch.device('mps'),load_model_gpu=lambda m:None,get_free_memory=lambda d:8*1024**3,throw_exception_if_processing_interrupted=lambda:interrupts.append(1))
    pre=NS(compute_target_size=lambda *a:(14,14),preprocess_image=lambda x,**kw:x.movedim(-1,1))
    module=NS(mm=mm,da3_preprocess=pre)
    model=NS(model=NS(diffusion_model=Diff()))
    image=torch.arange(5.).view(5,1,1,1).expand(5,14,14,3).clone()
    def stock(*a):raise AssertionError('unexpected fallback')
    depth,conf,sky=p._da3_run(stock,module,model,image,504)
    assert calls==[2,2,1] and len(interrupts)==3
    assert conf is None and torch.equal(depth,image[...,0]) and torch.equal(sky,image[...,1])
    mm.get_free_memory=lambda d:1024**3
    assert p._da3_run(lambda *a:'low-memory fallback',module,model,image,504)=='low-memory fallback'
