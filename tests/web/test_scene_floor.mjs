// node tests/web/test_scene_floor.mjs — room occluder + diagnostics of web/3da/sceneSelection.js (0.2.0), no browser.
// The 0.1.10 infinite floor plane (y = -0.01) is gone: the occluder is the location room mesh (plate_mesh.obj).
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

// A standing 'body': a 0.3 x 1.6 x 0.3 m box of points, origin at the feet.
function body(name, h=1.6, w=0.3){ const pts=[];
  for(let i=0;i<=6;i++)for(let j=0;j<=6;j++)for(let k=0;k<=16;k++) pts.push(-w/2+w*i/6, h*k/16, -w/2+w*j/6);
  const g=new Geo(); g.setAttribute('position',new Attr(new Float32Array(pts),3)); const o=new Mesh(g,new Mat()); o.name=name; return o; }
const root=new Obj('root'); const b=body('3DA_body'); const p=body('3DA_prop_0',0.3,0.1);
root.children.push(b,p); b.position.set(0,0,-3); p.position.set(0.155,0.9,-3);   // prop touches the body: held
// Room mesh of the location (plate world, Z up): floor in front of the camera + back wall, OBJ quads.
function grid(o,a,c,n=20){const v=[],f=[];for(let i=0;i<=n;i++)for(let j=0;j<=n;j++)v.push([0,1,2].map(k=>o[k]+a[k]*i/n+c[k]*j/n));
  for(let i=0;i<n;i++)for(let j=0;j<n;j++)f.push([i*(n+1)+j,i*(n+1)+j+1,(i+1)*(n+1)+j+1,(i+1)*(n+1)+j]);return {v,f};}
function obj(parts){let out='',base=1;for(const {v,f} of parts){out+=v.map(x=>'v '+x.join(' ')).join('\n')+'\n';
  out+=f.map(q=>'f '+q.map(i=>i+base).join(' ')).join('\n')+'\n';base+=v.length;}return out;}
const ROOM=obj([grid([-4,1,0],[8,0,0],[0,7,0]),grid([-4,8,0],[8,0,0],[0,0,3])]);
let fetched;
globalThis.fetch=async url=>{fetched=String(url);return {ok:true,status:200,text:async()=>ROOM};};
const scene={children:[],add(m){this.children.push(m);},remove(m){this.children=this.children.filter(x=>x!==m);}};
const made=[];
globalThis.document={baseURI:'http://comfy.local/',createElement:()=>{const e={style:{},append(){},appendChild(){},add(){},addEventListener(){},setAttribute(){},options:[{},{},{}],remove(){}};made.push(e);return e;}};
globalThis.Option=function(t,v){this.text=t;this.value=v;};
globalThis.requestAnimationFrame=f=>f();
const listeners={};
const view={getModelManager:()=>({currentModel:root}),getSceneManager:()=>({scene}),domElement:{parentElement:{appendChild(){}}},
  getGizmoManager:()=>({transformControls:{addEventListener:(k,f)=>(listeners[k]??=[]).push(f),removeEventListener(){}},getMode:()=>'translate',setupForModel(){}}),
  setGizmoEnabled(){},setGizmoMode(){},forceRender(){}};
const t=0.25; // camera 1.4 m high, pitched down ~14 deg (three.js Y up)
const camera={position:{x:0,y:1.4,z:0},target:{x:0,y:1.4-Math.sin(t),z:-Math.cos(t)},customUp:{x:0,y:Math.cos(t),z:-Math.sin(t)},fov:45};
const node={properties:{'3DA Scene':{camera,raster:[1344,768],room_mesh:'3d/3da_scene/room_test.obj',
  entities:[{id:'body',name:'3DA_body',kind:'body',label:'Персонаж'},{id:'prop_0',name:'3DA_prop_0',kind:'prop',label:'Предмет 1'}]}}};
const status=()=>made.filter(e=>e.className==='threeda-scene-status').at(-1).textContent;
const sel=sceneSelection(view,node);
assert.match(status(),/Загружаю геометрию комнаты/);
for(let i=0;i<5;i++) await new Promise(r=>setTimeout(r,0));
assert.match(fetched,/api\/view/); assert.match(fetched,/filename=room_test\.obj/); assert.match(fetched,/subfolder=3d%2F3da_scene/);
assert.ok(!scene.children.some(m=>m.name==='3DA_floor_occluder'),'no infinite 0.1.10 floor plane');
const occ=scene.children.find(m=>m.name==='3DA_room_occluder');
assert.ok(occ && occ.material.colorWrite===false && occ.material.depthWrite===true && occ.renderOrder===-1);
assert.equal(occ.geometry.index.itemSize,1);
assert.equal(status(),'Геометрия и рост: ок');
assert.equal(made.find(e=>e.className==='threeda-mode-selector').options[2].text,'Масштаб (равномерно)');
// edit: 10 cm into the floor and x1.26 (the 13.11 case, room-relative, not y=0)
b.position.set(0,-0.1,-3); b.scale.set(1.262,1.262,1.262); p.position.set(0.25,1.0,-3);
listeners.mouseUp.forEach(f=>f());
assert.match(status(),/Персонаж: внутри пола на \d+ см/);
assert.match(status(),/масштаб ×1\.26/);
assert.doesNotMatch(status(),/Предмет 1: нет опоры/,'held prop has support');
// floating 20 cm
b.position.set(0,0.2,-3); b.scale.set(1,1,1); p.position.set(0.155,1.1,-3);
listeners.mouseUp.forEach(f=>f());
assert.match(status(),/Персонаж: нет опоры — висит на 20 см/);
sel.dispose();
assert.equal(scene.children.length,0,'occluder removed on dispose');
// Old export without room_mesh: honest message, no fake floor.
delete node.properties['3DA Scene'].room_mesh;
const sel2=sceneSelection(view,node);
assert.match(status(),/без геометрии комнаты/); assert.equal(scene.children.length,0);
sel2.dispose();
console.log('ok');
