const assert=require('assert');
const fs=require('fs');
const path=require('path');
const {chromium}=require('playwright');
const output=process.argv[2]||'data/ui-checks/calibration';
(async()=>{
 fs.mkdirSync(output,{recursive:true});
 const config=JSON.parse(require('child_process').execFileSync(process.env.OPENGAZELINK_TEST_PYTHON || (process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python'),
   ['-c','import json; from dataclasses import asdict; from opengazelink_pc.config import ProviderConfig; print(json.dumps(asdict(ProviderConfig())))'],{encoding:'utf8'}));
 const status={config,engine:{},geometry:{configured:true},artifacts:{models:{},datasets:{}},calibration:{}};
 const browser=await chromium.launch({headless:true,channel:process.env.PLAYWRIGHT_CHANNEL || undefined});
 try {
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{window.EventSource=class{close(){}};
    const timeout=window.setTimeout;window.setTimeout=(fn,ms,...args)=>timeout(fn,ms===1000?30:ms,...args);
  });
  const mock=structuredClone(status);mock.engine.input={ready:true,error:''};mock.engine.tracking=false;
  mock.geometry.configured=true;mock.config.geometry_configured=true;mock.calibration={active:false,phase:'',state:'idle'};
  mock.artifacts.models.tasks_conditioned_video={ready:true,compatible:true};
  let starts=0,finishes=0,cancels=0,pauses=0,resumes=0,reviews=0,serverState='idle',plan=[],redoId=null;
  const batches=[],discarded=new Set();
  await page.route('http://127.0.0.1:8765/**',async route=>{
   const request=route.request(),pathname=new URL(request.url()).pathname;
   if(!pathname.startsWith('/api/')) {
    const file=path.join(process.cwd(),'web',pathname==='/'?'index.html':pathname.slice(1));
    if(fs.existsSync(file))await route.fulfill({body:fs.readFileSync(file),contentType:pathname.endsWith('.js')?'application/javascript':pathname.endsWith('.css')?'text/css':'text/html'});
    else await route.fulfill({status:404,body:''});return;
   }
   const body=request.postDataJSON()||{};let result={ok:true};
   if(pathname==='/api/status')result=mock;
   else if(pathname==='/api/config'){Object.assign(mock.config,body);result={ok:true,config:mock.config};}
   else if(pathname==='/api/video/clock')result.pc_ms=performance.now();
   else if(pathname==='/api/calibration/unified/start'){starts++;discarded.clear();plan=body.plan;serverState='collecting';mock.calibration={active:true,purpose:'unified',state:'collecting',phase:'unified_capture'};}
   else if(pathname==='/api/video/events'){assert.equal(serverState,'collecting');batches.push(body);}
   else if(pathname==='/api/video/pause'){pauses++;serverState='paused';if(body.discard_segment)discarded.add(body.discard_segment);}
   else if(pathname==='/api/video/resume'){assert.equal(serverState,'paused');resumes++;serverState='collecting';}
   else if(pathname==='/api/video/review'){
    reviews++;assert.equal(serverState,'paused');
    const events=batches.flatMap(b=>b.events).filter(e=>!discarded.has(e.capture_segment)&&e.phase!=='pause');
    const seen=new Set(events.filter(e=>e.trial_complete).map(e=>e.trial_id));
    if(!redoId&&seen.size>=plan.length)redoId=plan[0].trial_id;
    if(redoId&&reviews===1)seen.delete(redoId);
    const missing=plan.filter(s=>!seen.has(s.trial_id)).map(s=>s.trial_id);
    result={ready:!missing.length,missing_trials:missing,completed:plan.length-missing.length,total:plan.length};
   } else if(pathname==='/api/video/finish'){assert.equal(serverState,'paused');finishes++;serverState='training';mock.calibration={active:true,state:'training',purpose:'unified',phase:'unified_training'};result.directory='isolated-ui-test';}
   else if(pathname==='/api/calibration/cancel'){cancels++;serverState='idle';mock.calibration={active:false,state:'cancelled'};}
   await route.fulfill({json:result});
  });
  await page.goto('http://127.0.0.1:8765');
  await page.locator('[data-page="calibration"]').click();
  await page.waitForFunction(()=>!document.getElementById('startVideoButton').disabled);
  await page.evaluate(()=>{
    const original=window.UnifiedPlan;
    window.UnifiedPlan={...original,
      spatialPlan:(seed,options)=>original.spatialPlan(seed,options).map(t=>({...t,duration:t.duration/12,settle_ms:t.settle_ms/12,end_hold_ms:t.end_hold_ms/12})),
      dragProgress:(step,current,desired,dt)=>Math.min(desired,current+dt/350)};
  });
  // Opening the preparation screen does not start capture or train.
  await page.locator('#startVideoButton').click();await page.waitForTimeout(250);
  await page.screenshot({path:output+'/desktop-prepare.png'});
  assert.equal(starts,0);assert.equal(finishes,0);
   await page.locator('#resumeVideoButton').click();
   await page.evaluate(()=>{
     window.__driveSpatial=setInterval(()=>{
       const target=document.getElementById('videoTarget'),message=document.getElementById('videoMessage').textContent;
       if(target.hidden)return;
       const rect=document.getElementById('videoOverlay').getBoundingClientRect();
       const id=message.split(' · ')[0];
       if(message.includes('静态 · 待点击')) target.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,button:0,pointerId:1,clientX:rect.width/2,clientY:rect.height/2}));
       else if(message.includes('直线 · 待拖动')||message.includes('直线 · 采样中')) {
         target.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,button:0,pointerId:1,clientX:rect.width/2,clientY:rect.height/2}));
         const end=document.getElementById('videoRailEnd');
         const ex=Number(end.getAttribute('cx'))/100*rect.width,ey=Number(end.getAttribute('cy'))/100*rect.height;
         target.dispatchEvent(new PointerEvent('pointermove',{bubbles:true,pointerId:1,clientX:ex,clientY:ey}));
         target.dispatchEvent(new PointerEvent('pointerup',{bubbles:true,pointerId:1,clientX:ex,clientY:ey}));
       }
     },40);
   });
   await page.waitForFunction(()=>!document.getElementById('pauseVideoButton').hidden).catch(async error=>{
     console.log({message:await page.locator('#videoMessage').textContent(), status:await page.locator('#videoStatus').textContent(),
       toast:await page.locator('#toast').textContent(), starts, errors});throw error;
   });
  await page.screenshot({path:output+'/desktop-sampling.png'});
  const corners=await page.evaluate(()=>{
    const target=document.getElementById('videoTarget'),style=target.getAttribute('style');
    const results=[];
    for(const x of [18,innerWidth-18])for(const y of [18,innerHeight-18]) {
      target.style.left=`${x}px`;target.style.top=`${y}px`;
      const rect=target.getBoundingClientRect();
      results.push({visible:rect.left>=0&&rect.top>=0&&rect.right<=innerWidth&&rect.bottom<=innerHeight,
        clickable:document.elementFromPoint(x,y)===target});
    }
    if(style===null)target.removeAttribute('style');else target.setAttribute('style',style);
    return results;
  });
  assert(corners.every(c=>c.visible&&c.clickable));
  await page.waitForTimeout(40);
  await page.keyboard.press('Space');
  await page.waitForFunction(()=>!document.getElementById('resumeVideoButton').disabled);
  assert.equal(pauses,1);assert.equal(discarded.size,1);
  const n=batches.flatMap(b=>b.events).length;
  await page.waitForTimeout(350);assert.equal(batches.flatMap(b=>b.events).length,n);
  await page.locator('#resumeVideoButton').click();
  // All twelve lines advance without group pauses; only missing data asks for a retry.
  for(let retry=0;retry<1;retry++) {
    await page.waitForFunction(()=>!document.getElementById('videoControls').hidden&&!document.getElementById('resumeVideoButton').disabled&&/本组完成|需要补采|采集检查通过/.test(document.getElementById('videoMessage').textContent),{},{timeout:30000});
    const priorResumes=resumes;await page.waitForTimeout(150);assert.equal(resumes,priorResumes);assert.equal(finishes,0);
    await page.locator('#resumeVideoButton').click();
  }
  await page.waitForFunction(()=>document.getElementById('videoOverlay').hidden);
  await page.evaluate(()=>clearInterval(window.__driveSpatial));
  assert.equal(finishes,1);assert.equal(starts,1);assert.equal(cancels,0);
  assert.equal(reviews,2);assert.equal(pauses,3);
  const events=batches.flatMap(b=>b.events);
  assert(plan.every(s=>s.plan_version===10&&!s.head_pose));
  const viewport=await page.evaluate(()=>({width:innerWidth,height:innerHeight}));
  assert(plan.every(s=>s.viewport_width===viewport.width&&s.viewport_height===viewport.height));
  assert(events.every(e=>!/头|head/i.test(e.head_cue.text)));
  assert(events.every((e,i)=>!i||e.pc_ms>events[i-1].pc_ms));
  assert.equal(new Set(events.map(e=>e.trial_id)).size,plan.length);
  assert.deepEqual(new Set(events.map(e=>e.split)),new Set(['train','test']));
  assert.equal(await page.locator('#cancelUnifiedButton').count(),0);
  assert.equal(await page.locator('#regionMetrics').count(),0);
  assert(events.some(e=>e.rail));assert(events.some(e=>e.phase==='pause'));
  assert.deepEqual(errors,[]);
  mock.artifacts.models.tasks_conditioned_video.created_at='2026-09-18T01:23:45Z';
  mock.calibration={active:true,state:'training',phase:'unified_spatial',progress:{stage:'spatial',completed:7,total:30}};
  await page.evaluate(()=>refreshStatus());
  await page.waitForFunction(()=>document.getElementById('trainingProgress').value===7);
  assert.match(await page.locator('#videoStatus').textContent(),/7\/30/);
  assert.match(await page.locator('#activeModelInfo').textContent(),/2026/);
  await page.screenshot({path:output+'/training-progress.png'});
  mock.calibration={active:false,state:'complete',phase:'unified_complete'};
  await page.evaluate(()=>refreshStatus());
  await page.waitForFunction(()=>document.getElementById('trainingProgress').hidden);
  await page.screenshot({path:output+'/model-status.png'});
  mock.calibration={active:false,state:'idle'};await page.evaluate(()=>refreshStatus());
  await page.locator('#startVideoButton').click();await page.locator('#cancelVideoButton').click();
  assert.equal(starts,1);assert.equal(cancels,0);
  // Stage two is a separate explicit start and advances without mouse dragging.
  await page.evaluate(()=>{
    const original=window.UnifiedPlan.eventPlan;
    window.UnifiedPlan.eventPlan=seed=>{
      const plan=original(seed).map(t=>({...t,duration:t.duration/12,changes:t.changes.map(v=>v/12)}));
      window.__eventSeconds=Math.ceil(plan.reduce((n,s)=>n+s.duration,0)/1000);
      return plan;
    };
  });
  await page.locator('#startEventButton').click();
  assert.equal(starts,1);
  assert((await page.locator('#videoProgress').textContent()).includes(`12 段 · 约 ${await page.evaluate(()=>window.__eventSeconds)} 秒`));
  await page.screenshot({path:output+'/event-prepare.png'});
  await page.locator('#resumeVideoButton').click();
  await page.waitForFunction(()=>document.getElementById('videoOverlay').hidden,{},{timeout:30000}).catch(async e=>{console.log({message:await page.locator('#videoMessage').textContent(),progress:await page.locator('#videoProgress').textContent(),starts,finishes,reviews,serverState,errors,last:batches.at(-1)?.events.at(-1)});throw e;});
  assert.equal(starts,2);assert.equal(finishes,2);assert.equal(cancels,0);
  assert.equal(plan.length,12);assert(plan.every(s=>s.calibration_stage==='events_v1'&&s.plan_version===9));
  const eventFrames=batches.flatMap(b=>b.events).filter(e=>e.calibration_stage==='events_v1');
  assert.equal(new Set(eventFrames.map(e=>e.trial_id)).size,12);
  assert(eventFrames.some(e=>e.phase==='jump'));
  assert.deepEqual(errors,[]);
  const result={starts,finishes,cancels,pauses,resumes,reviews,discardedSegments:[...discarded],spatialTrialCount:12,eventTrialCount:plan.length,events:events.length,errors,
    notice:'Source UI tested with isolated mocked APIs and accelerated trials. No production capture or training was started.'};
  fs.writeFileSync(output+'/ui-check.json',JSON.stringify(result,null,2));console.log(result);
 } finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
