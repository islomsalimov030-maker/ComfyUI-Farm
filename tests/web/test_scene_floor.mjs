// node tests/web/test_scene_floor.mjs — floor occluder + diagnostics of web/3da/sceneSelection.js, no browser.
import assert from 'node:assert/strict';
import { sceneSelection } from '../../web/3da/sceneSelection.js';

class V3 { constructor(x=0,y=0,z=0){this.x=x;this.y=y;this.z=z;} set(x,y,z){this.x=x;this.y=y;this.z=z;return this;}
  clone(){return new V3(this.x,this.y,this.z);} copy(v){return this.set(v.x,v.y,v.z);} multiplyScalar(s){return this.set(this.x*s,this.y*s,this.z*s);} }
class Attr { constructor(array,itemSize){this.array=array;this.itemSize=itemSize;} }
class Geo { constructor(){this.attributes={};} setAttribute(k,a){this.attributes[k]=a;} setIndex(i){this.index=i;} dispose(){this.disposed=true;} }
class Mat { dispose(){this.disposed=true;} }
class Obj { constructor(name){this.name=name;this.children=[];this.position=new V3();this.scale=new V3(1,1,1);
    this.quaternion={x:0,y:0,z:0,w:1,set(){},clone(){return this;}};this.matrixWorld={elements:[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]};}
  updateMatrixWorld(){const s=this.scale.x,p=this.position;this.matrixWorld.elements=[s,0,0,0,0,s,0,0,0,0,s,0,p.x,p.y,p.z,1];
    for(const c of this.children){c.matrixWorld=this.matrixWorld;}}
  traverse(f){f(this);this.children.forEach(c=>c.traverse(f));} getObjectByName(n){let r;this.traverse(o=>{if(o.name===n)r=o;});return r;}
  getWorldPosition(v){return v;} getWorldQuaternion(q){return q;} getWorldScale(v){return v.copy(this.scale);} }
class Mesh extends Obj { constructor(geometry,material){super('mesh');this.isMesh=true;this.geometry=geometry;this.material=material;} }

function body(name, ys){ const g=new Geo(); g.setAttribute('position',new Attr(new Float32Array(ys.flatMap(y=>[0,y,0])),3));
  const o=new Mesh(g,new Mat()); o.name=name; return o; }
const root=new Obj('root'); const b=body('3DA_body',[-0.18,0.53]); const p=body('3DA_prop_0',[0,0.1]);
root.children.push(b,p); b.position.set(0,0.146,-2.8);
const scene={children:[],add(m){this.children.push(m);},remove(m){this.children=this.children.filter(x=>x!==m);}};
const made=[];
globalThis.document={createElement:()=>{const e={style:{},append(){},appendChild(){},add(){},addEventListener(){},setAttribute(){},options:[{},{},{}],remove(){}};made.push(e);return e;}};
globalThis.Option=function(t,v){this.text=t;this.value=v;};
globalThis.requestAnimationFrame=f=>f();
const listeners={};
const view={getModelManager:()=>({currentModel:root}),getSceneManager:()=>({scene}),domElement:{parentElement:{appendChild(){}}},
  getGizmoManager:()=>({transformControls:{addEventListener:(k,f)=>(listeners[k]??=[]).push(f),removeEventListener(){}},getMode:()=>'translate',setupForModel(){}}),
  setGizmoEnabled(){},setGizmoMode(){},forceRender(){}};
const node={properties:{'3DA Scene':{entities:[{id:'body',name:'3DA_body',kind:'body',label:'Персонаж'},{id:'prop_0',name:'3DA_prop_0',kind:'prop',label:'Предмет 1'}]}}};
const sel=sceneSelection(view,node);
const occ=scene.children.find(m=>m.name==='3DA_floor_occluder');
assert.ok(occ && occ.material.colorWrite===false && occ.material.depthWrite===true && occ.renderOrder===-1);
const status=()=>made.find(e=>e.className==='threeda-scene-status').textContent;
assert.equal(status(),'Пол и рост: ок');
assert.equal(made.find(e=>e.className==='threeda-mode-selector').options[2].text,'Масштаб (равномерно)');
// edit: user's 13.11 transform — 40 cm down, ×1.26
b.position.set(0.38,-0.26,-2.36); b.scale.set(1.262,1.262,1.262);
listeners.mouseUp.forEach(f=>f());
assert.match(status(),/Персонаж: ниже пола на 49 см/);
assert.match(status(),/масштаб ×1\.26/);
sel.dispose();
assert.equal(scene.children.length,0,'occluder removed on dispose');
console.log('ok');
