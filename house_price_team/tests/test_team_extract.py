from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,sys,tempfile,unittest
import numpy as np
import pandas as pd
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import team_extract as e
SOURCES=e.read(ROOT/'MODEL_SOURCES.json')

class Guard:
 def check(self):pass

class Response:
 def __init__(self,body,status=200,headers=None):self.body=body;self.status=status;self.headers=headers or {};self.pos=0
 def getcode(self):return self.status
 def __enter__(self):return self
 def __exit__(self,*_):pass
 def read(self,n):
  a=self.body[self.pos:self.pos+n];self.pos+=len(a);return a

class Tests(unittest.TestCase):
 def test_readout_skips_registers_and_respects_mask(self):
  spec=SOURCES['models']['register_large336'];tokens=torch.zeros(1,581,1024)
  tokens[:,0]=7;tokens[:,1:5]=1000000;tokens[:,5:]=3
  mask=torch.ones(1,1,336,336)
  actual=e.readout(SimpleNamespace(last_hidden_state=tokens),mask,spec)
  self.assertTrue(torch.equal(actual[:,:1024],torch.full((1,1024),7.)))
  self.assertTrue(torch.equal(actual[:,1024:],torch.full((1,1024),3.)))
  tokens[:,5]=999;mask[:,:,:14,:14]=0
  self.assertTrue(torch.equal(e.readout(SimpleNamespace(last_hidden_state=tokens),mask,spec)[:,1024:],torch.full((1,1024),3.)))
 def test_letterbox_zeros_padding_and_preserves_valid_area(self):
  image=Image.fromarray(np.full((10,20,3),255,np.uint8))
  x,mask=e.preprocess(image,SOURCES['models']['l336'])
  self.assertEqual(tuple(x.shape),(3,336,336));self.assertEqual(mask.sum().item(),336*168)
  self.assertTrue(torch.equal(x[:,mask[0]==0],torch.zeros_like(x[:,mask[0]==0])))
  self.assertTrue(torch.equal(mask[:,84:252,:],torch.ones(1,168,336)))
 def test_siglip_preprocessing_matches_torchvision(self):
  from torchvision import transforms
  from torchvision.transforms import InterpolationMode
  image=Image.fromarray(np.random.default_rng(1).integers(0,256,(137,219,3),dtype=np.uint8))
  x,mask=e.preprocess(image,SOURCES['models']['full'])
  expected=transforms.Compose([transforms.Resize((256,256),interpolation=InterpolationMode.BILINEAR),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])(image)
  self.assertIsNone(mask);torch.testing.assert_close(x,expected,atol=0,rtol=0)
 def test_no_price_or_fold_columns_read(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'rows.csv'
   pd.DataFrame({'key':['train:a.jpg'],'imageid':['a.jpg'],'split':['train'],'image_sha256':['a'*64],'price':['not-a-price'],'outer_fold':['private']}).to_csv(path,index=False)
   rows=e.read_rows(path)
   self.assertEqual(set(rows.columns),{'key','imageid','split','image_sha256'})
 def test_cache_resume_detects_committed_corruption(self):
  with tempfile.TemporaryDirectory() as d:
   cache=Path(d);values,state=e.open_cache(cache,'sig',4,2,Guard())
   e.commit_cache(cache,values,state,0,np.ones((2,2),np.float32))
   del values
   values,state=e.open_cache(cache,'sig',4,2,Guard());self.assertEqual(state['completed_rows'],2)
   values[0,0]=2;values.flush();del values
   with self.assertRaises(ValueError):e.open_cache(cache,'sig',4,2,Guard())
 def test_cache_refuses_overlap_and_changed_binding(self):
  with tempfile.TemporaryDirectory() as d:
   cache=Path(d);values,state=e.open_cache(cache,'sig',4,2,Guard())
   e.commit_cache(cache,values,state,0,np.ones((2,2),np.float32))
   with self.assertRaises(ValueError):e.commit_cache(cache,values,state,0,np.ones((2,2),np.float32))
   with self.assertRaises(ValueError):e.open_cache(cache,'other',4,2,Guard())
 def test_download_resume_and_checksum(self):
  body=b'abcdefgh';expected={'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'model';p.with_name('model.partial').write_bytes(body[:3]);requests=[]
   def opener(request,**kw):requests.append(request);return Response(body[3:],206,{'Content-Range':'bytes 3-7/8'})
   e.download(['https://example.test/model'],p,expected,Guard(),opener)
   self.assertEqual(p.read_bytes(),body);self.assertEqual(requests[0].headers['Range'],'bytes=3-')
 def test_download_mirror_fallback_and_no_overwrite(self):
  body=b'abcdefgh';expected={'bytes':8,'sha256':hashlib.sha256(body).hexdigest()}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'model'
   def opener(request,**kw):
    if 'bad.test' in request.full_url:raise OSError('network unavailable')
    return Response(body)
   e.download(['https://bad.test/model','https://good.test/model'],p,expected,Guard(),opener)
   self.assertEqual(p.read_bytes(),body)
   p.write_bytes(b'original')
   with self.assertRaises(ValueError):e.download(['https://good.test/model'],p,expected,Guard(),opener)
   self.assertEqual(p.read_bytes(),b'original')
 def test_wrong_range_is_never_appended(self):
  body=b'abcdefgh';expected={'bytes':8,'sha256':hashlib.sha256(body).hexdigest()}
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'model';partial=p.with_name('model.partial');partial.write_bytes(body[:3])
   def opener(request,**kw):return Response(body[4:],206,{'Content-Range':'bytes 4-7/8'})
   with self.assertRaises(RuntimeError):e.download(['https://example.test/model'],p,expected,Guard(),opener)
   self.assertEqual(partial.read_bytes(),body[:3]);self.assertFalse(p.exists())
 def test_reference_id_hash_alignment(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'ref.npz';np.savez(p,keys=['train:b','train:a'],image_sha256=['b'*64,'a'*64],features=np.asarray([[2.,3.],[4.,5.]],np.float32))
   rows=pd.DataFrame({'key':['train:a','train:b'],'image_sha256':['a'*64,'b'*64]})
   np.testing.assert_array_equal(e.reference_features(p,rows,2),[[4,5],[2,3]])
   rows.loc[0,'image_sha256']='c'*64
   with self.assertRaises(ValueError):e.reference_features(p,rows,2)
 def test_export_preserves_existing_feature_and_provenance(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'new.npz';rows=pd.DataFrame({'key':['train:a'],'image_sha256':['a'*64]});v=np.ones((1,2),np.float32)
   a=e.export_features(p,rows,v,'sig',{},Guard());original=p.read_bytes()
   self.assertEqual(a['feature_sha256'],hashlib.sha256(original).hexdigest())
   e.export_features(p,rows,v,'sig',{},Guard())
   with self.assertRaises(FileExistsError):e.export_features(p,rows,v,'other',{},Guard())
   self.assertEqual(p.read_bytes(),original)
 def test_stop_guard(self):
  with tempfile.TemporaryDirectory() as d:
   g=e.Guard(d);g.check();(Path(d)/'STOP_REQUESTED.txt').write_text('')
   with self.assertRaises(e.PlannedStop):g.check()
   (Path(d)/'STOP_REQUESTED.txt').unlink();(Path(d)/'STOP_REQUESTED').write_text('')
   with self.assertRaises(e.PlannedStop):g.check()

if __name__=='__main__':unittest.main(verbosity=2)
