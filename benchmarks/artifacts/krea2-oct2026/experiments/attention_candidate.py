from pathlib import Path
import functools,re,sys,torch
ROOT=Path(__file__).parent
MODE='baseline';stats={};original=None;qc=None

def set_mode(mode):
 global MODE
 MODE=mode;stats.clear()

@functools.lru_cache(maxsize=8)
def library(hadamard,bhsd=False,input_bhsd=False):
 source=(ROOT/'attention-generator'/('hadamard.metal' if hadamard else 'plain.metal')).read_text()
 if input_bhsd:
  source=source.replace('src_vec[index / width]', 'src_vec[(input_offset + (head * sequence + start + row) * 128 + dim) / width]')
  source=source.replace('src4[index / 4]', 'src4[(batch * batch_stride + (head * sequence + row) * 128 + vec_dim * 4) / 4]')
  vstart=source.index('kernel void quantize_v(')
  vpref,vbody=source[:vstart],source[vstart:]
  vbody=vbody.replace('(batch * batch_stride + (head * sequence + row) * 128 + vec_dim * 4)', '(batch * QUANTIZE_V_BATCH_STRIDE + (head * QUANTIZE_KV_SEQUENCE + row) * 128 + vec_dim * 4)')
  vpref=vpref.replace('(row * QUANTIZE_KV_HEADS + head) * 128', '(head * QUANTIZE_KV_SEQUENCE + row) * 128')
  source=vpref+vbody
 if bhsd:
  source=source.replace('auto O = O_buf + tgid.x * (16 * K_Hq) + tgid.y * 128;', 'auto O = O_buf + tgid.y * R * 128 + tgid.x * (16 * 128);')
  source=re.sub(r'(O\[[^\n]+idx\[1\] \*) K_Hq', r'\1 128',source)
 values={'QUANTIZE_Q_HEADS':48,'QUANTIZE_KV_HEADS':12,'QUANTIZE_Q_TILE_SIZE':16,'QUANTIZE_KV_TILE_SIZE':64,'V_mean_batch_stride':12*128}
 source=re.sub(r'constant uint (\w+) \[\[function_constant\(\d+\)\]\];',lambda m:f'constant uint {m[1]} = {values.get(m[1],0)};',source)
 bindings={}
 for match in reversed(list(re.finditer(r'kernel void (\w+)\(',source))):
  start,end=match.start(),source.index('{',match.start());signature=source[start:end]
  indices=[int(m[1]) for m in re.finditer(r'\[\[buffer\((\d+)\)\]\]',signature)];bindings[match[1]]=indices
  remap={old:new for new,old in enumerate(indices)}
  signature=re.sub(r'\[\[buffer\((\d+)\)\]\]',lambda m:f'[[buffer({remap[int(m[1])]})]]',signature)
  source=source[:start]+signature+source[end:]
 (ROOT/('compiled-hadamard.metal' if hadamard else 'compiled-plain.metal')).write_text(source)
 return torch.mps.compile_shader(source),bindings

@functools.lru_cache(maxsize=8)
def dimensions(r,c):
 return torch.tensor([r,c,r*48*128,c*12*128,c*12*128,r*48*128,0,0,0,0,48*((r+15)//16),12*((c+63)//64),12*((c+63)//64)],dtype=torch.int32,device='mps')

def attention(q,k,v,hadamard=False,bhsd=False,input_bhsd=False):
 assert q.shape[0]==1 and q.shape[1]==48 and k.shape[1]==12 and q.shape[-1]==128
 assert q.dtype==k.dtype==v.dtype==torch.bfloat16
 r,c=q.shape[2],k.shape[2];lib,bindings=library(hadamard,bhsd,input_bhsd)
 if input_bhsd:
  assert all(x.is_contiguous() for x in (q,k,v))
 else:q,k,v=[x.transpose(1,2).contiguous() for x in (q,k,v)]
 qi=torch.empty_like(q,dtype=torch.int8);ki=torch.empty_like(k,dtype=torch.int8);vi=torch.empty_like(v,dtype=torch.int8)
 qs=torch.empty((48,(r+15)//16),device='mps');ks=torch.empty((12,(c+63)//64),device='mps');vs=torch.empty_like(ks)
 vm=torch.empty((12,128),device='mps');km=torch.empty_like(vm)
 chunks=(c+511)//512;partial=torch.empty((2 if hadamard else 1,12,chunks,32,4),device='mps')
 correction=torch.empty((48,r),device='mps');out=torch.empty((1,48,r,128) if bhsd else q.shape,dtype=q.dtype,device='mps');lse=torch.empty((48,r),device='mps')
 dims=dimensions(r,c)
 def dispatch(name,buf,groups,threads):
  getattr(lib,name)(*[buf[i] for i in bindings[name]],threads=tuple(g*t for g,t in zip(groups,threads)),group_size=threads)
 # With centering, both means are computed before quantizing Q and K.
 meanbuf={0:v,1:vm,2:k,3:km,4:partial,21:dims}
 dispatch('compute_v_mean',meanbuf,(chunks,12*(2 if hadamard else 1),1),(256,1,1))
 if chunks>1:dispatch('finalize_v_mean',meanbuf,(1,12*(2 if hadamard else 1),1),(32,1,1))
 dispatch('quantize_q',{0:q,1:qi,2:qs,3:km,4:correction,21:dims},((r+15)//16,48,1),(128,1,1))
 dispatch('quantize_k',{0:k,1:ki,2:ks,3:km,21:dims},((c+63)//64,12,1),(256,1,1))
 dispatch('quantize_v',{0:v,1:vi,2:vs,3:vm,21:dims},((c+63)//64,12,1),(256,1,1))
 dispatch('int8_attention',{0:qi,1:ki,2:vi,3:out,4:lse,10:qs,11:ks,12:vs,14:vm,21:dims,22:correction},(((r+63)//64)*48,1,1),(128,1,1))
 return out if bhsd else out.transpose(1,2)

@functools.lru_cache(maxsize=2)
def exact_library(bq):
 source=qc._source().replace('bfloat, 64, 32, 128, 4, 1,','bfloat, %d, 32, 128, %d, 1,'%(bq,bq//16))
 return torch.mps.compile_shader(source)
def exact_attention(q,k,v,bq=32):
 out=torch.empty(q.shape,device='mps',dtype=q.dtype)
 p=qc._parameters(tuple(q.shape),tuple(k.shape),q.stride(),k.stride(),v.stride(),128**-.5)
 exact_library(bq).custom_attn(q,k,v,out,p,p,p,threads=(((q.shape[2]+bq-1)//bq)*(bq//16)*32,q.shape[1],q.shape[0]),group_size=((bq//16)*32,1,1))
 return out

def install():
 global original,qc
 patches={n.rsplit('.',1)[-1]:m for n,m in list(sys.modules.items()) if '._patches.' in n}
 route=patches['torch_attention'];original=route.sdpa;qc=patches['torch_mpp_qcache'];captured=False
 def sdpa(q,k,v,**kw):
  nonlocal captured
  if q.ndim!=4 or q.shape[1]!=48 or k.shape[1]!=12 or q.dtype!=torch.bfloat16:return original(q,k,v,**kw)
  if MODE=='capture' and not captured:
   captured=True;torch.save([x.cpu() for x in (q,k,v)],ROOT/'attention-inputs.pt')
  if MODE not in ('plain','hadamard','plain_bhsd','hadamard_bhsd','hadamard_direct','q32','q32_verify'):return original(q,k,v,**kw)
  result=exact_attention(q,k,v) if MODE.startswith('q32') else attention(q,k,v,MODE.startswith('hadamard'),MODE.endswith('_bhsd') or MODE=='hadamard_direct',MODE=='hadamard_direct')
  if MODE=='q32_verify':
   ref=original(q,k,v,**kw)
   assert torch.equal(result.cpu().view(torch.uint8),ref.cpu().view(torch.uint8)), 'Q32 byte divergence'
   stats['exact_checks']=stats.get('exact_checks',0)+1
  stats['attention_calls']=stats.get('attention_calls',0)+1
  return result if not kw.get('output_bshd',False) else result.transpose(1,2)
 route.sdpa=sdpa
