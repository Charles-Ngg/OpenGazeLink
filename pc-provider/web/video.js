/* One user-controlled capture for spatial calibration and motion prediction. */
(() => {
  let run=null;
  const overlay=$('videoOverlay'),target=$('videoTarget');
  const controls=$('videoControls'),resumeButton=$('resumeVideoButton'),pauseButton=$('pauseVideoButton');
  function message(text){$('videoMessage').textContent=text;}
  function panel(text,label=tr('继续采集','Continue capture')) {
    message(text);controls.hidden=false;target.hidden=true;$('videoRail').setAttribute('hidden','');
    resumeButton.textContent=label;resumeButton.disabled=false;pauseButton.hidden=true;
  }
  async function synchronize() {
    let best=null;const probes=[];
    for(let i=0;i<7;i++) {
      const a=performance.now(),result=await post('/api/video/clock'),b=performance.now();
      probes.push({browser_send_ms:a,browser_receive_ms:b,pc_ms:result.pc_ms});
      if(!best||b-a<best.rtt)best={rtt:b-a,offset:result.pc_ms-(a+b)/2};
    }
    if(best.rtt>40)throw new Error(tr('页面通信延迟过大，请在电脑本机控制页重试','Page latency is too high. Retry in the control page on this PC.'));
    return {...best,probes};
  }
  async function flush(s) {
    if(s.sending)return s.sending;
    if(!s.events.length)return;
    // The plan is already persisted by /start. Repeating every trial in every
    // telemetry request used tens of megabytes without adding recoverability.
    const batch=s.pendingBatch||{batch_id:`${s.id}-${s.batchIndex++}`,events:s.events.slice(0,120)};
    s.pendingBatch=batch;s.sending=post('/api/video/events',batch);
    try{await s.sending;s.events.splice(0,batch.events.length);s.pendingBatch=null;}
    finally{s.sending=null;}
  }
  async function drain(s) {if(s.sending)await s.sending;while(s.events.length)await flush(s);}
  function currentPoint(s) {
    const step=s.steps[s.index],line=step.motion_profile==='line';
    if(step.calibration_stage==='events_v1')return {...window.UnifiedPlan.sample(step,s.elapsed),motionAge:0};
    const position=line?step.point.map((v,i)=>v+(step.end[i]-v)*s.dragProgress):[...step.point];
    const phase=s.stepState==='ready'?'pause':s.stepState==='moving'?'pursuit':'anchor';
    return {position,phase,rail:phase==='pursuit'?{a:step.point,b:step.end}:null,motionAge:s.motionAge};
  }
  function sample(s,now,dt,paused=false,force=false) {
    const step=s.steps[s.index],point=currentPoint(s);
    if(now<=s.lastTelemetryAt)return point;
    const event={browser_ms:now,pc_ms:now+s.clock.offset,sync_rtt_ms:s.clock.rtt,
      x:point.position[0],y:point.position[1],block:step.block,trial_id:step.trial_id,split:step.split,
      phase:paused?'pause':point.phase,visible:!document.hidden,motion_profile:step.motion_profile,
      condition:window.UnifiedPlan.headCue(step,s.dragProgress,s.stepState).condition,
      head_cue:window.UnifiedPlan.headCue(step,s.dragProgress,s.stepState),rail:paused?null:point.rail,motion_age_ms:point.motionAge,
      trial_age_ms:s.elapsed,step_age_ms:s.elapsed,capture_segment:s.segment,
      calibration_stage:step.calibration_stage,drag_progress:s.dragProgress,
      trial_complete:s.stepState==='complete',
      display_frame_id:s.displayFrame++,display_dt_ms:dt,display_clock:'raf_submission',
      viewport_width:overlay.clientWidth,viewport_height:overlay.clientHeight};
    s.lastTelemetryAt=now;s.events.push(event);s.localEvents.push(event);return point;
  }
  function enterStep(s) {
    s.elapsed=0;s.previous=performance.now();s.lastTelemetryAt=-Infinity;s.segment++;
    const step=s.steps[s.index];
    s.stepState=step.calibration_stage==='events_v1'?'event':'ready';s.dragProgress=0;s.dragDesired=0;s.motionAge=0;s.stateAge=0;s.dragPointer=null;
    target.classList.remove('collecting');overlay.classList.remove('video-dragging');
    const line=step.motion_profile==='line';$('videoRail').toggleAttribute('hidden',!line);
    if(line) {
      for(const [key,value] of Object.entries({x1:step.point[0]*100,y1:step.point[1]*100,x2:step.end[0]*100,y2:step.end[1]*100}))$('videoRailLine').setAttribute(key,value);
      $('videoRailEnd').setAttribute('cx',step.end[0]*100);$('videoRailEnd').setAttribute('cy',step.end[1]*100);
    }
    overlay.style.background='#6b6b6b';
    s.interactions.push({kind:'trial_start',browser_ms:s.previous,trial_id:step.trial_id,capture_segment:s.segment});
  }
  function draw(now) {
    const s=run;if(!s||s.paused||s.finishing)return;
    const dt=Math.max(0,now-s.previous);s.previous=now;
    if(dt>250){pause(tr('显示出现中断，已暂停；继续后重做当前小段。','Display interrupted. Paused; resume to repeat the current sequence.')).catch(()=>{});return;}
    const step=s.steps[s.index];
    if(s.stepState!=='ready') {s.elapsed+=dt;s.stateAge+=dt;}
    if(s.stepState==='event'&&s.stateAge>=step.duration) {s.stepState='complete';s.stateAge=500;}
    if(s.stepState==='collecting'&&s.stateAge>=step.duration) {s.stepState='complete';s.stateAge=0;}
    if(s.stepState==='settling'&&s.stateAge>=step.settle_ms) {s.stepState='moving';s.stateAge=0;}
    if(s.stepState==='moving') {
      const before=s.dragProgress;
      s.dragProgress=window.UnifiedPlan.dragProgress(step,before,s.dragDesired,dt);
      s.motionAge=s.dragProgress>before+1e-7?s.motionAge+dt:0;
      if(s.dragProgress>=.999) {s.dragProgress=1;s.stepState='landing';s.stateAge=0;}
    }
    if(s.stepState==='landing'&&s.stateAge>=step.end_hold_ms) {s.stepState='complete';s.stateAge=0;}
    target.classList.toggle('collecting',s.stepState!=='ready'&&s.stepState!=='complete');
    const point=sample(s,now,dt);
    target.style.left=`${point.position[0]*100}%`;target.style.top=`${point.position[1]*100}%`;
    const labels={ready:tr('直线 · 待拖动','Line · drag to begin'),collecting:tr('采样中','Capturing'),settling:tr('起点 · 保持注视','Start · hold gaze'),moving:tr('沿直线拖动','Drag along the line'),landing:tr('终点 · 保持注视','End · hold gaze'),complete:tr('本段完成','Sequence complete'),event:tr('看向蓝点，跳转后继续注视','Look at the dot and follow each jump')};
    message(`${tr(step.group,s.eventStage?'Saccades & fixations':`Group ${step.rest_block+1}/3`)} · ${labels[s.stepState]} · ${window.UnifiedPlan.headCue(step,s.dragProgress,s.stepState,language).text}`);
    $('videoProgress').textContent=`${s.retry?tr('补采','Retry'):tr('进度','Progress')} ${s.index+1}/${s.steps.length}${step.motion_profile==='line'?` · ${Math.round(s.dragProgress*100)}%`:''}`;
    if(s.events.length>240){pause(tr('发送暂时积压，已暂停。','Upload is behind; capture paused.')).catch(()=>{});return;}
    if(s.stepState==='complete'&&s.stateAge>=(s.eventStage?500:0)) {
      s.completed.add(step.trial_id);
      const next=s.index+1;
      if(next>=s.steps.length) {
        pause(tr('本组完成，正在检查有效数据…','Group complete. Checking captured data…'),true).then(()=>review(s,next)).catch(e=>failure(s,e));return;
      }
      s.index=next;enterStep(s);
    }
    s.raf=requestAnimationFrame(draw);
  }
  async function pause(reason=tr('已暂停；继续后从当前小段起点重做。','Paused. Resuming will restart this sequence.'),boundary=false) {
    const s=run;if(!s||!s.started||s.paused||s.finishing)return;
    s.paused=true;s.busy=true;cancelAnimationFrame(s.raf);clearInterval(s.timer);
    s.dragPointer=null;overlay.classList.remove('video-dragging');
    panel(reason);resumeButton.disabled=true;
    sample(s,performance.now(),0,true);
    s.interactions.push({kind:boundary?'block_complete':'pause',browser_ms:performance.now(),capture_segment:s.segment});
    try {
      await drain(s);
      await post('/api/video/pause',boundary?{}:{discard_segment:s.segment});
      s.serverPaused=true;
      if(!boundary)s.elapsed=0;
      panel(reason);
    } catch(error){failure(s,error);throw error;}
    finally{s.busy=false;}
  }
  async function review(s,next) {
    if(run!==s||s.finishing)return;
    s.busy=true;resumeButton.disabled=true;
    try {
      const quality=await post('/api/video/review');
      s.quality=quality;
      const failed=quality.missing_trials.filter(id=>s.completed.has(id));
      if(next<s.steps.length) {
        s.index=next;
        panel(tr(`本组完成，可休息。已通过 ${quality.completed}/${quality.total} 段检查${failed.length?`；${failed.length} 段需要补采`:''}。`,`Group complete. Take a break. ${quality.completed}/${quality.total} sequences passed${failed.length?`; ${failed.length} need a retry`:''}.`),tr('开始下一组','Start next group'));
      } else if(!quality.ready) {
        s.steps=s.plan.filter(step=>quality.missing_trials.includes(step.trial_id));s.index=0;s.retry=true;
        panel(tr(`还有 ${s.steps.length} 段需要补采。请确认眼睛清晰可见；已通过的内容会保留。`,`${s.steps.length} sequences need a retry. Keep your eyes clearly visible. Passed sequences are kept.`),tr('补采缺失小段','Retry missing sequences'));
      } else {
        s.ready=true;
        panel(tr(`阶段完成，共 ${quality.total} 段。正在保存并${s.eventStage?'检查眼跳与注视':'训练位置模型'}…`,`Stage complete: ${quality.total} sequences. Saving and ${s.eventStage?'evaluating saccades and fixations':'training the position model'}…`),tr('正在保存','Saving'));
        await finish(s);
      }
    } finally{s.busy=false;resumeButton.disabled=false;}
  }
  async function continueRun() {
    const s=run;if(!s||s.busy||s.finishing)return;
    if(s.ready){await finish(s);return;}
    const token=++s.countdownToken;
    s.busy=true;resumeButton.disabled=true;
    try {
      if(document.fullscreenElement!==overlay)await overlay.requestFullscreen();
      for(let count=3;count>0;count--) {
        message(tr(`${count} 秒后开始 · 看向蓝点出现的位置`,`Starting in ${count} · Look toward the blue dot`));
        await new Promise(resolve=>setTimeout(resolve,1000));
        if(run!==s||s.finishing)return;
        if(token!==s.countdownToken||document.hidden||document.fullscreenElement!==overlay) {
          panel(tr('准备已暂停，返回全屏后点击继续。','Paused. Return to fullscreen and continue.'));return;
        }
      }
      if(!s.started) {
        await saveConfig();
        s.clock=await synchronize();
        if(run!==s||s.finishing)return;
        // Build pixel-sized edge margins from the actual fullscreen viewport.
        if(!s.eventStage)s.plan=s.steps=window.UnifiedPlan.spatialPlan(s.plan[0].seed,{width:overlay.clientWidth,height:overlay.clientHeight});
        await post('/api/calibration/unified/start',{plan:s.plan});s.started=true;
        if(run!==s||s.finishing){await post('/api/calibration/cancel');return;}
      } else {
        await post('/api/video/resume');s.serverPaused=false;
        if(run!==s||s.finishing){await post('/api/calibration/cancel');return;}
      }
      if(document.hidden||document.fullscreenElement!==overlay) {
        await post('/api/video/pause');s.serverPaused=true;panel(tr('准备已暂停，返回全屏后点击继续。','Paused. Return to fullscreen and continue.'));return;
      }
      s.paused=false;s.busy=false;enterStep(s);
      controls.hidden=true;target.hidden=false;pauseButton.hidden=false;
      s.timer=setInterval(()=>{if(run===s&&!s.paused&&!s.finishing)flush(s).catch(e=>failure(s,e));},100);
      s.raf=requestAnimationFrame(draw);
    } catch(error){failure(s,error);}
    finally{s.busy=false;}
  }
  function saveTelemetry(s,error) {
    const blob=new Blob([JSON.stringify({clock:s.clock,plan:s.plan,events:s.localEvents,interactions:s.interactions,pending:s.pendingBatch,error})],{type:'application/json'});
    const link=document.createElement('a'),url=URL.createObjectURL(blob);
    link.href=url;link.download=`unified-telemetry-${s.id}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);
  }
  async function failure(s,error) {
    if(run!==s||s.finishing)return;
    await exit(error.message,true);
  }
  async function closeOverlay(s) {
    if(run===s)run=null;actionBusy=false;overlay.hidden=true;
    if(document.fullscreenElement===overlay)await document.exitFullscreen().catch(()=>{});
    refreshStatus();
  }
  async function exit(reason='',failed=false) {
    const s=run;if(!s||s.finishing)return;
    s.finishing=true;cancelAnimationFrame(s.raf);clearInterval(s.timer);
    try {
      if(s.started) {
        if(!s.serverPaused) {
          if(!s.paused)sample(s,performance.now(),0,true);
          await drain(s);
        }
        await post('/api/calibration/cancel');
      }
      $('videoStatus').textContent=reason||tr('已结束本次采集，原始数据保留。','Capture ended. Original data is kept.');
      if(failed)saveTelemetry(s,reason);
    } catch(error){saveTelemetry(s,error.message);$('videoStatus').textContent=error.message;await post('/api/calibration/cancel').catch(()=>{});}
    finally{await closeOverlay(s);}
  }
  async function finish(s) {
    s.busy=true;resumeButton.disabled=true;
    try {
      const result=await post('/api/video/finish');s.finishing=true;
      $('videoStatus').textContent=tr('第一阶段已保存，正在训练位置模型','Stage 1 saved. Training the position model.');
      if(s.eventStage)$('videoStatus').textContent=tr('第二阶段已保存，正在检查注视稳定与眼跳落点','Stage 2 saved. Evaluating stability and landing behavior.');
      await closeOverlay(s);
    } catch(error){panel(tr(`尚未完成保存：${error.message}`,`Save is incomplete: ${error.message}`),tr('重试保存','Retry save'));}
    finally{s.busy=false;}
  }
  async function prepare(eventStage=false) {
    if(actionBusy||run||state?.calibration?.active)return;
    configPayload();
    actionBusy=true;if(state)renderStatus(state);
    const seed=crypto.getRandomValues(new Uint32Array(1))[0],plan=eventStage?window.UnifiedPlan.eventPlan(seed):window.UnifiedPlan.spatialPlan(seed);
    run={eventStage,id:Date.now().toString(36),plan,steps:plan,index:0,segment:0,elapsed:0,displayFrame:0,
      events:[],localEvents:[],interactions:[],batchIndex:0,sending:null,paused:true,started:false,countdownToken:0,
      lastTelemetryAt:-Infinity,completed:new Set()};
    overlay.hidden=false;$('videoRail').setAttribute('hidden','');$('videoProgress').textContent=tr('第一阶段 · 12 条直线 · 连续采集','Stage 1 · 12 lines · Continuous capture');
    const eventSeconds=Math.ceil(plan.reduce((sum,step)=>sum+step.duration,0)/1000);
    panel(eventStage?tr(`第二阶段 · 约 ${eventSeconds} 秒，眼睛跟随跳转点`,`Stage 2 · About ${eventSeconds} seconds. Follow the jumping dot.`):tr('第一阶段 · 注视并沿线拖动蓝点，覆盖四角、边缘和中央','Stage 1 · Follow and drag the blue dot across the corners, edges and center.'),eventStage?tr('开始第二阶段','Start stage 2'):tr('开始第一阶段','Start stage 1'));
    if(eventStage)$('videoProgress').textContent=tr(`第二阶段 · ${plan.length} 段 · 约 ${eventSeconds} 秒`,`Stage 2 · ${plan.length} sequences · About ${eventSeconds} seconds`);
    try{await overlay.requestFullscreen();}catch(error){message(tr(`进入全屏后开始采集：${error.message}`,`Enter fullscreen to capture: ${error.message}`));}
  }
  $('startVideoButton').addEventListener('click',()=>prepare(false).catch(error=>{actionBusy=false;toast(error.message);refreshStatus();}));
  $('startEventButton').addEventListener('click',()=>prepare(true).catch(error=>{actionBusy=false;toast(error.message);refreshStatus();}));
  target.addEventListener('pointerdown',event=>{
    const s=run;if(!s||s.paused||s.finishing||event.button!==0)return;
    if(s.stepState==='ready') {
      s.stepState=s.steps[s.index].motion_profile==='anchor'?'collecting':'settling';
      s.elapsed=0;s.stateAge=0;s.previous=performance.now();
      s.interactions.push({kind:'sample_start',browser_ms:s.previous,trial_id:s.steps[s.index].trial_id});
    }
    if(s.steps[s.index].motion_profile==='line'&&['settling','moving'].includes(s.stepState)) {
      s.dragPointer=event.pointerId;target.setPointerCapture(event.pointerId);overlay.classList.add('video-dragging');
    }
    event.preventDefault();
  });
  target.addEventListener('pointermove',event=>{
    const s=run;if(!s||s.paused||s.dragPointer!==event.pointerId)return;
    const step=s.steps[s.index],rect=overlay.getBoundingClientRect();
    const dx=(step.end[0]-step.point[0])*rect.width,dy=(step.end[1]-step.point[1])*rect.height;
    const x=event.clientX-rect.left-step.point[0]*rect.width,y=event.clientY-rect.top-step.point[1]*rect.height;
    s.dragDesired=Math.max(s.dragProgress,Math.min(1,(x*dx+y*dy)/(dx*dx+dy*dy)));
  });
  function releasePointer(event) {
    const s=run;if(!s||s.dragPointer!==event.pointerId)return;
    s.dragPointer=null;overlay.classList.remove('video-dragging');
  }
  target.addEventListener('pointerup',releasePointer);
  target.addEventListener('pointercancel',releasePointer);
  window.addEventListener('resize',()=>{if(run&&!run.paused)pause(tr('显示尺寸改变，已暂停。','Display size changed; capture paused.')).catch(()=>{});});
  resumeButton.addEventListener('click',continueRun);
  pauseButton.addEventListener('click',()=>pause().catch(()=>{}));
  $('cancelVideoButton').addEventListener('click',()=>exit());
  document.addEventListener('keydown',event=>{
    if(!run)return;
    if(event.code==='Space'){event.preventDefault();if(run.paused)continueRun();else pause().catch(()=>{});}
    if(event.code==='Escape'&&run.paused&&run.busy){run.countdownToken++;return;}
    if(event.code==='Escape'&&!run.paused){event.preventDefault();pause(tr('已退出全屏并暂停；继续时重做当前小段。','Left fullscreen; paused. Resume to repeat this sequence.')).catch(()=>{});}
  });
  document.addEventListener('fullscreenchange',()=>{if(run&&run.paused&&document.fullscreenElement!==overlay)run.countdownToken++;if(run&&!run.paused&&document.fullscreenElement!==overlay)pause(tr('已退出全屏并暂停；继续时重做当前小段。','Left fullscreen; paused. Resume to repeat this sequence.')).catch(()=>{});});
  document.addEventListener('visibilitychange',()=>{if(run&&run.paused&&document.hidden)run.countdownToken++;if(run&&!run.paused&&document.hidden)pause(tr('页面切到后台，已暂停；返回后可继续。','Page is in the background; paused. Return to continue.')).catch(()=>{});});
})();
