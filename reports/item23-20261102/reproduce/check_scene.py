"""Browser test for wheel geometry in arbitrary Three.js-generated pages."""
import argparse
import os
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument("html", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
vendor = Path(os.environ.get("ITEM23_THREE_VENDOR", str(Path.home() / "dsh/dsh2/test3/vendor/three")))
args.out.parent.mkdir(parents=True,exist_ok=True)
hook = """
globalThis.__sceneQA = {scenes:new Set(),camera:null};
const qaAdd=Object3D.prototype.add;
Object3D.prototype.add=function(...objects){
 if(this.isScene)globalThis.__sceneQA.scenes.add(this);
 return qaAdd.apply(this,objects);
};
globalThis.__sceneQA.sample=function(){
 const wheels=[];
 for(const scene of this.scenes){
  scene.updateMatrixWorld(true);
  scene.traverse(mesh=>{
   if(mesh.geometry?.type==='TorusGeometry'){
    const radius=mesh.geometry.parameters.radius;
    if(radius>.2)wheels.push({mesh,radius,position:mesh.getWorldPosition(new Vector3()),normal:new Vector3(0,0,1).transformDirection(mesh.matrixWorld)});
   }
  });
 }
 wheels.sort((a,b)=>b.radius-a.radius);
 const selected=[];
 for(const wheel of wheels){
  if(selected.every(w=>w.position.distanceTo(wheel.position)>wheel.radius))selected.push(wheel);
  if(selected.length===2)break;
 }
 if(selected.length!==2)return {assessed:false,reason:'Could not identify two distinct torus wheels'};
 const line=selected[1].position.clone().sub(selected[0].position).normalize();
 const axle=new Vector3(0,1,0).cross(line).normalize();
 const spokes=selected.map(w=>{
  let count=0,maxRadius=0;
  w.mesh.parent.traverse(mesh=>{
   const p=mesh.geometry?.parameters;
   if(mesh.geometry?.type!=='CylinderGeometry'||p.height<w.radius*.5||p.radiusTop>.03)return;
   count++;
   for(const sign of [-1,1]){
    const end=new Vector3(0,sign*p.height/2,0).applyMatrix4(mesh.matrixWorld);
    maxRadius=Math.max(maxRadius,end.distanceTo(w.position));
   }
  });
  return {count,maxRadius,limit:w.radius*1.08,contained:count>0&&maxRadius<=w.radius*1.08};
 });
 const movingPlanes=[];
 for(const scene of this.scenes)scene.traverse(mesh=>{
  const p=mesh.geometry?.parameters;
  if(mesh.geometry?.type==='PlaneGeometry'&&p.width*p.height<5)movingPlanes.push({id:mesh.uuid,position:mesh.getWorldPosition(new Vector3()).toArray()});
 });
 return {assessed:true,wheels:selected.map(w=>({radius:w.radius,position:w.position.toArray(),normal:w.normal.toArray(),alignment:Math.abs(w.normal.dot(axle)),vertical:Math.abs(w.normal.y)<.2})),spokes,movingPlanes,travelAxis:line.toArray(),rendered:!!this.camera};
};
"""
errors=[]
result={"input":str(args.html),"pass":False}
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path="/opt/google/chrome/chrome",headless=True,args=["--use-angle=swiftshader","--enable-unsafe-swiftshader"])
    page=browser.new_page(viewport={"width":1100,"height":750})
    page.add_init_script("window.__frames=[]; window.__qaTime=1000; performance.now=()=>window.__qaTime; window.requestAnimationFrame=cb=>(window.__frames.push(cb),window.__frames.length);")
    page.on("pageerror",lambda e:errors.append(str(e)))
    page.route("https://pelican.test/",lambda route:route.fulfill(body=args.html.read_text(),content_type="text/html"))
    def modules(route):
        url=route.request.url
        rel="three.module.js" if "/build/three.module" in url else "addons/"+url.split("/examples/jsm/")[-1]
        file=vendor/rel
        data=file.read_text()
        if rel=="three.module.js":
            marker="this.render = function ( scene, camera ) {"
            if marker not in data:raise RuntimeError("renderer hook not found")
            data=data.replace(marker,marker+"\n globalThis.__sceneQA.camera=camera;")
            data+="\n"+hook
        route.fulfill(body=data,content_type="text/javascript")
    page.route("https://cdn.jsdelivr.net/npm/three@0.160.0/**",modules)
    page.route("https://unpkg.com/three@0.160.0/**",modules)
    try:
        page.goto("https://pelican.test/",wait_until="networkidle",timeout=15000)
        page.wait_for_function("window.__sceneQA?.camera",timeout=5000)
        result["initial"]=page.evaluate("()=>window.__sceneQA.sample()")
        page.evaluate("()=>{for(let i=0;i<60;i++){window.__qaTime+=1000/60;const cb=window.__frames.shift();if(cb)cb(window.__qaTime);}}")
        result["after_one_second"]=page.evaluate("()=>window.__sceneQA.sample()")
        result["pass"]=all(s.get("assessed") and s.get("rendered") and all(w["alignment"]>.95 and w["vertical"] for w in s["wheels"]) and all(w["contained"] for w in s["spokes"]) for s in [result["initial"],result["after_one_second"]])
        before={o["id"]:o["position"] for o in result["initial"].get("movingPlanes",[])}
        import math
        alignments=[]
        axis=result["after_one_second"].get("travelAxis",[])
        for obj in result["after_one_second"].get("movingPlanes",[]):
            if obj["id"] not in before:continue
            delta=[b-a for a,b in zip(before[obj["id"]],obj["position"])]
            length=math.sqrt(sum(v*v for v in delta))
            if length>.01:alignments.append(abs(sum(a*b for a,b in zip(axis,delta)))/length)
        result["scenery_alignment"]=alignments
        if alignments:result["pass"]=result["pass"] and min(alignments)>.95
        page.evaluate("()=>{for(let i=0;i<30;i++){window.__qaTime+=1000/30;const cb=window.__frames.shift();if(cb)cb(window.__qaTime);}}")
        result["after_second_at_30fps"]=page.evaluate("()=>window.__sceneQA.sample()")
        def median_motion(first, second):
            import statistics
            positions={o["id"]:o["position"] for o in first.get("movingPlanes",[])}
            distances=[]
            for obj in second.get("movingPlanes",[]):
                if obj["id"] in positions:
                    distances.append(math.sqrt(sum((a-b)**2 for a,b in zip(obj["position"],positions[obj["id"]]))))
            return statistics.median(distances) if distances else None
        first=median_motion(result["initial"],result["after_one_second"])
        second=median_motion(result["after_one_second"],result["after_second_at_30fps"])
        result["frame_rate_motion"]={"distance_60fps":first,"distance_30fps":second}
        if first is not None:
            result["pass"]=result["pass"] and first>.01 and abs(second/first-1)<.05
    except Exception as e:
        result["error"]=str(e)
    page.screenshot(path=str(args.out.with_suffix(".png")))
    browser.close()
result["browser_errors"]=errors
result["pass"]=result["pass"] and not errors
args.out.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
raise SystemExit(0 if result["pass"] else 1)
