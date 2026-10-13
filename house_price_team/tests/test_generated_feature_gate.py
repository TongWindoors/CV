"""Small-dimensional synthetic provenance fixtures; no GPU execution is claimed."""
from pathlib import Path
import hashlib,json,sys,tempfile,unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));import team_extract as e

class GeneratedGateTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.folder=self.root/'generated_features';self.folder.mkdir()
  (self.root/'data').mkdir();(self.root/'assets/features').mkdir(parents=True)
  (self.root/'team_extract.py').write_bytes((ROOT/'team_extract.py').read_bytes())
  self.rows=pd.DataFrame({'imageid':[f'{i}.jpg' for i in range(11000)],'split':['train']*8000+['test']*3000,'image_sha256':['0'*64]*11000})
  self.rows['key']=self.rows['split']+':'+self.rows.imageid
  self.rows['price']=np.r_[np.ones(8000),np.full(3000,np.nan)]
  self.rows.to_csv(self.root/'data/ROWS.csv',index=False)
  self.rows=self.rows[['key','imageid','split','image_sha256','price']]
  sources={'models':{}}
  for name in e.NAMES:
   sources['models'][name]={'repo':'fixture/'+name,'revision':'a'*40,'files':{'config.json':{'sha256':'b'*64}},'accepted_local_weights':[{'sha256':'c'*64}],
    'feature_dim':2,'readout':'locked fixture readout','preprocessing':'locked fixture transform'}
  e.save(self.root/'MODEL_SOURCES.json',sources);files={}
  for name in e.NAMES:
   path=self.root/'assets/features'/(name+'.npz')
   np.savez(path,keys=self.rows.key.to_numpy(dtype=str),image_sha256=self.rows.image_sha256.to_numpy(dtype=str),features=np.ones((11000,2),np.float32))
   files['assets/features/'+name+'.npz']={'bytes':path.stat().st_size,'sha256':e.sha(path)}
  path=self.root/'data/ROWS.csv';files['data/ROWS.csv']={'bytes':path.stat().st_size,'sha256':e.sha(path)}
  e.save(self.root/'ASSET_MANIFEST.json',{'files':files})
  self.status={'status':'complete','probe_only':False,'branches':{},'backbones_finetuned':False,'prices_read':False,
    'sources_sha256':e.sha(self.root/'MODEL_SOURCES.json'),'extractor_sha256':e.sha(self.root/'team_extract.py')}
  hardware={'device':'cuda','precision':'FP32','name':'synthetic fixture; no GPU run'}
  for name in e.NAMES:
   spec=sources['models'][name];model={'repo':spec['repo'],'revision':spec['revision'],'weights_sha256':'c'*64,'config_sha256':'b'*64,'directory':'fixture'}
   binding={'format':1,'model':{k:v for k,v in model.items() if k!='directory'},'spec':spec,'extractor_sha256':self.status['extractor_sha256'],
    'rows':self.rows[['key','imageid','split','image_sha256']].to_dict('records'),'hardware':hardware,'precision':'FP32'}
   signature=hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
   (self.folder/(name+'.npz')).write_bytes((self.root/'assets/features'/(name+'.npz')).read_bytes())
   meta={'status':'complete','n':11000,'weights_updated':False,'price_labels_read':False,'model':model,'hardware':hardware,
    'preprocessing':spec['preprocessing'],'readout':spec['readout'],'signature':signature,'feature_sha256':e.sha(self.folder/(name+'.npz')),'difference':{'within_fp32_tolerance':True}}
   e.save(self.folder/(name+'.json'),meta);self.status['branches'][name]=meta
  e.save(self.folder/'EXTRACTION_STATUS.json',self.status)
 def tearDown(self):self.temp.cleanup()
 def test_accepts_complete_bound_numerical_reproduction(self):
  result=e.validate_generated_features(self.root,self.folder,self.rows)
  self.assertEqual(result['status'],'passed');self.assertEqual(len(result['branches']),4)
 def test_rejects_probe_or_missing_branch(self):
  self.status['probe_only']=True;e.save(self.folder/'EXTRACTION_STATUS.json',self.status)
  with self.assertRaises(ValueError):e.validate_generated_features(self.root,self.folder,self.rows)
  self.status['probe_only']=False;self.status['branches'].pop('full');e.save(self.folder/'EXTRACTION_STATUS.json',self.status)
  with self.assertRaises(ValueError):e.validate_generated_features(self.root,self.folder,self.rows)
 def test_rejects_tampered_npz(self):
  p=self.folder/'full.npz';p.write_bytes(p.read_bytes()+b'modified')
  with self.assertRaises(ValueError):e.validate_generated_features(self.root,self.folder,self.rows)
 def test_rejects_changed_preprocessing_and_canonical_rows(self):
  p=self.folder/'full.json';meta=e.read(p);meta['preprocessing']='different';e.save(p,meta)
  with self.assertRaises(ValueError):e.validate_generated_features(self.root,self.folder,self.rows)
  meta['preprocessing']='locked fixture transform';e.save(p,meta)
  rows=self.rows.copy();rows.loc[0,'image_sha256']='1'*64
  with self.assertRaises(ValueError):e.validate_generated_features(self.root,self.folder,rows)
 def test_rejects_reference_asset_substitution(self):
  p=self.root/'assets/features/full.npz';p.write_bytes(p.read_bytes()+b'substitution')
  with self.assertRaises(ValueError):e.validate_generated_features(self.root,self.folder,self.rows)

if __name__=='__main__':unittest.main(verbosity=2)
